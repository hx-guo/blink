//! 编号（管线 S4.5）：把一个目录版本的事件与登记表对上，发新编号、记变更、撤销。
//!
//! 编号 `yymmddAA`（展示为 `TGF yymmddAA`）：峰值的 UTC 日期加两位字母，按分配
//! 顺序 AA, AB, …, ZZ，每天最多 676 个。全局一套，不分卫星。**发布后永久占用**：
//! 名字、峰值、分配时间不改，撤销或合并的编号不回收、不复用、不改指向。
//! 设计见 docs/tgf-pipeline-design.md 第 3 节。
//!
//! 登记表目录（单独的 git 仓库）：
//!
//! * `names.csv`       name, peak_utc, assigned_at, assigned_by, status, status_reason,
//!   merged_into, status_changed_at —— 只追加行、只改 status 那几列
//! * `detections.csv`  name, instrument, start, stop, peak, false_positive_per_year,
//!   count, longitude, latitude, altitude_km, version_first, version_last
//! * `history.csv`     name, changed_at, version, field, old, new —— 只追加
//! * `public_instruments.txt`  一行一个仪器名；不在名单里的仪器不发编号
//!
//! 匹配：同一仪器峰值差 ≤ 1 ms 或时间窗重叠即同一事件；不同仪器之间峰值差
//! ≤ 星间距离/c + 1 ms。一个连通块里按（编号分配先后, 峰值差）贪心配对，于是
//! 拆分时编号跟着最近的那个走、合并时留最早分配的编号。
//!
//! 只给 `final` 事件发编号；`provisional` 事件可以对上已有编号（数据重发后回到
//! provisional 的情形），但不会因此撤销，也不发新号。
//!
//! 默认只打印报告，`--apply` 才写登记表。写之前再校验一遍不可改的列。
//! 实时运行用 `--apply-new-only`：只写不改动已发布编号状态的部分（沿用、新号、
//! 跨仪器探测），撤销、合并、恢复留给人工确认（设计文档第 3 节），有保留项时返回 true。

use chrono::{DateTime, NaiveDate, Utc};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

/// 同一仪器判为同一事件的峰值差（纳秒）
const SAME_EVENT_NS: i64 = 1_000_000;
/// 不同仪器之间在光行时差之外再放宽的量（纳秒）
const CROSS_SLACK_NS: i64 = 1_000_000;
const LIGHT_SPEED_KM_S: f64 = 299_792.458;
const EARTH_RADIUS_KM: f64 = 6371.0;

const NAMES_HEADER: &str =
    "name,peak_utc,assigned_at,assigned_by,status,status_reason,merged_into,status_changed_at";
const DETECTIONS_HEADER: &str = "name,instrument,start,stop,peak,false_positive_per_year,count,\
     longitude,latitude,altitude_km,version_first,version_last";
const HISTORY_HEADER: &str = "name,changed_at,version,field,old,new";

#[derive(Clone, Debug, PartialEq)]
struct NameRow {
    name: String,
    peak_utc: String,
    assigned_at: String,
    assigned_by: String,
    status: String,
    status_reason: String,
    merged_into: String,
    status_changed_at: String,
}

#[derive(Clone, Debug)]
struct Detection {
    name: String,
    instrument: String,
    start: String,
    stop: String,
    peak: String,
    fa: String,
    count: String,
    longitude: String,
    latitude: String,
    altitude_km: String,
    version_first: String,
    version_last: String,
}

/// 目录里的一个事件（catalog.csv 的一行）
#[derive(Clone, Debug)]
struct Event {
    final_: bool,
    start: String,
    stop: String,
    peak: String,
    fa: String,
    count: String,
    longitude: String,
    latitude: String,
    altitude_km: String,
}

fn ns(t: &str) -> i64 {
    t.parse::<DateTime<Utc>>()
        .unwrap_or_else(|e| panic!("bad time `{t}`: {e}"))
        .timestamp_nanos_opt()
        .expect("time in range")
}

/// 按表头读 CSV，返回逐行的列名 → 值。字段里不含逗号（这几张表的值都受控）。
fn read_table(path: &Path, header: &str) -> Vec<HashMap<String, String>> {
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut lines = content.lines();
    let columns: Vec<String> = match lines.next() {
        Some(h) => h.split(',').map(|c| c.trim().to_string()).collect(),
        None => return Vec::new(),
    };
    for required in header.split(',') {
        assert!(
            columns.iter().any(|c| c == required),
            "{} has no `{required}` column",
            path.display()
        );
    }
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            columns
                .iter()
                .cloned()
                .zip(l.split(',').map(str::to_string))
                .collect()
        })
        .collect()
}

fn get(row: &HashMap<String, String>, key: &str) -> String {
    row.get(key).cloned().unwrap_or_default()
}

fn write_atomic(path: &Path, content: &str) {
    let temp = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&temp, content).expect("failed to write registry file");
    fs::rename(&temp, path).expect("failed to rename registry file");
}

/// 字母序号 0..676 ↔ `AA`..`ZZ`
fn letters(index: usize) -> String {
    assert!(index < 676, "more than 676 TGFs in one day");
    let a = b'A' + (index / 26) as u8;
    let b = b'A' + (index % 26) as u8;
    String::from_utf8(vec![a, b]).unwrap()
}

fn letter_index(name: &str) -> usize {
    let b = name.as_bytes();
    (b[6] - b'A') as usize * 26 + (b[7] - b'A') as usize
}

/// 星下点 + 高度 → 地心直角坐标（km），球形地球足够算光行时差
fn ecef(lon: &str, lat: &str, alt_km: &str) -> Option<[f64; 3]> {
    let (lon, lat, alt): (f64, f64, f64) = (lon.parse().ok()?, lat.parse().ok()?, alt_km.parse().ok()?);
    let r = EARTH_RADIUS_KM + alt;
    let (lon, lat) = (lon.to_radians(), lat.to_radians());
    Some([r * lat.cos() * lon.cos(), r * lat.cos() * lon.sin(), r * lat.sin()])
}

fn light_time_ns(a: &[String; 3], b: &[String; 3]) -> Option<i64> {
    let p = ecef(&a[0], &a[1], &a[2])?;
    let q = ecef(&b[0], &b[1], &b[2])?;
    let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
    Some((d / LIGHT_SPEED_KM_S * 1e9) as i64)
}

fn same_instrument_match(d: &Detection, e: &Event) -> bool {
    let overlap = ns(&d.start) <= ns(&e.stop) && ns(&e.start) <= ns(&d.stop);
    overlap || (ns(&d.peak) - ns(&e.peak)).abs() <= SAME_EVENT_NS
}

/// 一次运行要做的全部改动
#[derive(Default, Debug)]
struct Plan {
    /// (编号, 事件下标)：对上的已有编号，含重新出现的已撤销编号
    matched: Vec<(String, usize)>,
    reinstated: Vec<String>,
    /// (编号, 原因)
    retracted: Vec<(String, String)>,
    /// (被并的编号, 保留的编号)
    merged: Vec<(String, String)>,
    /// (已有编号, 事件下标)：另一台仪器探测到的同一事件
    cross: Vec<(String, usize)>,
    /// (新编号, 事件下标)
    new: Vec<(String, usize)>,
    provisional_unnamed: usize,
}

struct Registry {
    names: Vec<NameRow>,
    detections: Vec<Detection>,
    public: BTreeSet<String>,
}

fn load_registry(dir: &Path) -> Registry {
    let names = read_table(&dir.join("names.csv"), NAMES_HEADER)
        .iter()
        .map(|r| NameRow {
            name: get(r, "name"),
            peak_utc: get(r, "peak_utc"),
            assigned_at: get(r, "assigned_at"),
            assigned_by: get(r, "assigned_by"),
            status: get(r, "status"),
            status_reason: get(r, "status_reason"),
            merged_into: get(r, "merged_into"),
            status_changed_at: get(r, "status_changed_at"),
        })
        .collect();
    let detections = read_table(&dir.join("detections.csv"), DETECTIONS_HEADER)
        .iter()
        .map(|r| Detection {
            name: get(r, "name"),
            instrument: get(r, "instrument"),
            start: get(r, "start"),
            stop: get(r, "stop"),
            peak: get(r, "peak"),
            fa: get(r, "false_positive_per_year"),
            count: get(r, "count"),
            longitude: get(r, "longitude"),
            latitude: get(r, "latitude"),
            altitude_km: get(r, "altitude_km"),
            version_first: get(r, "version_first"),
            version_last: get(r, "version_last"),
        })
        .collect();
    let public = fs::read_to_string(dir.join("public_instruments.txt"))
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    Registry {
        names,
        detections,
        public,
    }
}

fn plan(
    registry: &Registry,
    instrument: &str,
    events: &[Event],
    rejected: &[Rejected],
    range: (i64, i64),
) -> Plan {
    let mut plan = Plan::default();
    let by_name: HashMap<&str, &NameRow> =
        registry.names.iter().map(|n| (n.name.as_str(), n)).collect();

    // 本仪器、在目录数据范围内、编号未被合并的旧探测
    let own: Vec<&Detection> = registry
        .detections
        .iter()
        .filter(|d| d.instrument == instrument)
        .filter(|d| by_name.get(d.name.as_str()).is_some_and(|n| n.status != "merged"))
        .filter(|d| (range.0..range.1).contains(&ns(&d.peak)))
        .collect();

    // 边：旧探测 × 事件。按（编号分配先后, 峰值差）贪心配对
    let mut edges: Vec<(String, i64, usize, usize)> = Vec::new();
    for (k, d) in own.iter().enumerate() {
        for (i, e) in events.iter().enumerate() {
            if same_instrument_match(d, e) {
                let assigned = by_name[d.name.as_str()].assigned_at.clone();
                edges.push((assigned, (ns(&d.peak) - ns(&e.peak)).abs(), k, i));
            }
        }
    }
    edges.sort();
    let mut detection_paired = vec![false; own.len()];
    let mut event_name: Vec<Option<String>> = vec![None; events.len()];
    for (_, _, k, i) in &edges {
        if !detection_paired[*k] && event_name[*i].is_none() {
            detection_paired[*k] = true;
            event_name[*i] = Some(own[*k].name.clone());
            plan.matched.push((own[*k].name.clone(), *i));
            if by_name[own[*k].name.as_str()].status == "retracted" {
                plan.reinstated.push(own[*k].name.clone());
            }
        }
    }
    // 没配上的旧探测：有边 → 并进那个事件的编号；无边 → 撤销
    for (k, d) in own.iter().enumerate() {
        if detection_paired[k] {
            continue;
        }
        let into = edges
            .iter()
            .filter(|(_, _, kk, _)| *kk == k)
            .min_by_key(|(_, dt, _, _)| *dt)
            .and_then(|(_, _, _, i)| event_name[*i].clone());
        let status = by_name[d.name.as_str()].status.as_str();
        match into {
            Some(keep) if keep != d.name => plan.merged.push((d.name.clone(), keep)),
            Some(_) => {}
            None if status == "active" => {
                let reason = rejected
                    .iter()
                    .find(|(start, stop, _)| {
                        ns(&d.start) <= ns(stop) && ns(start) <= ns(&d.stop)
                    })
                    .map(|(_, _, reason)| format!("rejected:{reason}"))
                    .unwrap_or_else(|| "not_in_catalog".to_string());
                plan.retracted.push((d.name.clone(), reason));
            }
            None => {}
        }
    }

    // 没配上的事件：final 的先找别的仪器的同一事件，再发新号
    let others: Vec<&Detection> = registry
        .detections
        .iter()
        .filter(|d| d.instrument != instrument)
        .filter(|d| by_name.get(d.name.as_str()).is_some_and(|n| n.status == "active"))
        .collect();
    let mut next_letter: BTreeMap<String, usize> = BTreeMap::new();
    for n in &registry.names {
        let next = next_letter.entry(n.name[..6].to_string()).or_default();
        *next = (*next).max(letter_index(&n.name) + 1);
    }
    let mut unnamed: Vec<usize> = (0..events.len()).filter(|&i| event_name[i].is_none()).collect();
    unnamed.sort_by_key(|&i| ns(&events[i].peak));
    for i in unnamed {
        let e = &events[i];
        if !e.final_ {
            plan.provisional_unnamed += 1;
            continue;
        }
        let here = [e.longitude.clone(), e.latitude.clone(), e.altitude_km.clone()];
        let cross = others.iter().find(|d| {
            let there = [d.longitude.clone(), d.latitude.clone(), d.altitude_km.clone()];
            match light_time_ns(&here, &there) {
                Some(light) => (ns(&d.peak) - ns(&e.peak)).abs() <= light + CROSS_SLACK_NS,
                None => false,
            }
        });
        if let Some(d) = cross {
            plan.cross.push((d.name.clone(), i));
            continue;
        }
        let date = ns_date(ns(&e.peak));
        let next = next_letter.entry(date.clone()).or_default();
        plan.new.push((format!("{date}{}", letters(*next)), i));
        *next += 1;
    }
    plan
}

fn ns_date(t: i64) -> String {
    DateTime::from_timestamp_nanos(t).format("%y%m%d").to_string()
}

fn detection_of(name: &str, instrument: &str, e: &Event, version: &str) -> Detection {
    Detection {
        name: name.to_string(),
        instrument: instrument.to_string(),
        start: e.start.clone(),
        stop: e.stop.clone(),
        peak: e.peak.clone(),
        fa: e.fa.clone(),
        count: e.count.clone(),
        longitude: e.longitude.clone(),
        latitude: e.latitude.clone(),
        altitude_km: e.altitude_km.clone(),
        version_first: version.to_string(),
        version_last: version.to_string(),
    }
}

/// 把计划落到登记表上，返回 (names, detections, 新增的 history 行)
fn apply(
    registry: &Registry,
    plan: &Plan,
    instrument: &str,
    events: &[Event],
    version: &str,
    assigned_by: &str,
    now: &str,
) -> (Vec<NameRow>, Vec<Detection>, Vec<String>) {
    let mut names = registry.names.clone();
    let mut detections = registry.detections.clone();
    let mut history = Vec::new();
    let mut log = |name: &str, field: &str, old: &str, new: &str| {
        history.push(format!("{name},{now},{version},{field},{old},{new}"));
    };
    let mut set_status = |names: &mut Vec<NameRow>, name: &str, status: &str, reason: &str, into: &str| {
        let row = names.iter_mut().find(|n| n.name == name).expect("known name");
        log(name, "status", &row.status, status);
        row.status = status.to_string();
        row.status_reason = reason.to_string();
        row.merged_into = into.to_string();
        row.status_changed_at = now.to_string();
    };
    for name in &plan.reinstated {
        set_status(&mut names, name, "active", "reinstated", "");
    }
    for (name, reason) in &plan.retracted {
        set_status(&mut names, name, "retracted", reason, "");
    }
    for (name, keep) in &plan.merged {
        set_status(&mut names, name, "merged", "merged", keep);
    }

    let mut changes = Vec::new();
    for (name, i) in &plan.matched {
        let e = &events[*i];
        let d = detections
            .iter_mut()
            .find(|d| &d.name == name && d.instrument == instrument)
            .expect("matched detection exists");
        for (field, old, new) in [
            ("start", &mut d.start, &e.start),
            ("stop", &mut d.stop, &e.stop),
            ("peak", &mut d.peak, &e.peak),
            ("false_positive_per_year", &mut d.fa, &e.fa),
            ("count", &mut d.count, &e.count),
        ] {
            if old != new {
                changes.push((name.clone(), field, old.clone(), new.clone()));
                *old = new.clone();
            }
        }
        d.longitude = e.longitude.clone();
        d.latitude = e.latitude.clone();
        d.altitude_km = e.altitude_km.clone();
        d.version_last = version.to_string();
    }
    for (name, field, old, new) in changes {
        history.push(format!("{name},{now},{version},{instrument}:{field},{old},{new}"));
    }
    for (name, i) in &plan.cross {
        detections.push(detection_of(name, instrument, &events[*i], version));
        history.push(format!("{name},{now},{version},detection,,{instrument}"));
    }
    for (name, i) in &plan.new {
        names.push(NameRow {
            name: name.clone(),
            peak_utc: events[*i].peak.clone(),
            assigned_at: now.to_string(),
            assigned_by: assigned_by.to_string(),
            status: "active".into(),
            status_reason: String::new(),
            merged_into: String::new(),
            status_changed_at: String::new(),
        });
        detections.push(detection_of(name, instrument, &events[*i], version));
    }
    (names, detections, history)
}

/// 不可改的列与行数：只许追加，已有编号的名字、峰值、分配时间、分配者不变
fn verify_append_only(old: &[NameRow], new: &[NameRow]) {
    assert!(new.len() >= old.len(), "names.csv lost rows");
    for (a, b) in old.iter().zip(new) {
        assert!(
            a.name == b.name
                && a.peak_utc == b.peak_utc
                && a.assigned_at == b.assigned_at
                && a.assigned_by == b.assigned_by,
            "immutable columns of {} changed",
            a.name
        );
    }
    let mut seen = BTreeSet::new();
    for n in new {
        assert!(seen.insert(&n.name), "duplicated name {}", n.name);
    }
}

/// rejected.csv 的一行：(start, stop, reason)
type Rejected = (String, String, String);

fn read_catalog(dir: &Path) -> (String, (i64, i64), Vec<Event>, Vec<Rejected>) {
    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(dir.join("manifest.json")).expect("catalog dir has no manifest.json"),
    )
    .expect("bad manifest.json");
    let instrument = manifest["instrument"].as_str().expect("manifest.instrument").to_string();
    let day = |key: &str| -> NaiveDate {
        manifest["data"][key]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("manifest.data.{key} missing"))
    };
    let range = (
        day("first_day").and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp_nanos_opt().unwrap(),
        (day("last_day") + chrono::TimeDelta::days(1))
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap(),
    );
    let events = read_table(&dir.join("catalog.csv"), "status,start,stop,peak")
        .iter()
        .map(|r| Event {
            final_: get(r, "status") == "final",
            start: get(r, "start"),
            stop: get(r, "stop"),
            peak: get(r, "peak"),
            fa: get(r, "false_positive_per_year"),
            count: get(r, "count"),
            longitude: get(r, "longitude"),
            latitude: get(r, "latitude"),
            altitude_km: get(r, "altitude_km"),
        })
        .collect();
    // rejected.csv 只有 start；拿 start 当一个零宽窗
    let rejected = read_table(&dir.join("rejected.csv"), "start,reason")
        .iter()
        .map(|r| (get(r, "start"), get(r, "start"), get(r, "reason")))
        .collect();
    (instrument, range, events, rejected)
}

/// 写不写登记表
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Apply {
    /// 只打报告
    DryRun,
    /// 只写沿用、新号、跨仪器探测；撤销、合并、恢复留给人工
    NewOnly,
    /// 全部写入（人工确认过报告之后）
    All,
}

/// 返回是否有因 `Apply::NewOnly` 而留待人工确认的改动
pub fn cmd_name(catalog: &Path, registry_dir: &Path, version: &str, mode: Apply) -> bool {
    let (instrument, range, events, rejected) = read_catalog(catalog);
    let registry = load_registry(registry_dir);
    assert!(
        registry.public.contains(&instrument),
        "{instrument} is not in {}/public_instruments.txt: names are only given to public instruments",
        registry_dir.display()
    );

    let mut plan = plan(&registry, &instrument, &events, &rejected, range);
    let n_final = events.iter().filter(|e| e.final_).count();
    println!(
        "name {instrument} ({version}): {} events ({n_final} final); matched {} \
         (reinstated {}), new {}, other-instrument detections {}, retracted {}, merged {}, \
         provisional without a name {}",
        events.len(),
        plan.matched.len(),
        plan.reinstated.len(),
        plan.new.len(),
        plan.cross.len(),
        plan.retracted.len(),
        plan.merged.len(),
        plan.provisional_unnamed,
    );
    for name in &plan.reinstated {
        println!("  REINSTATE {name}");
    }
    for (name, reason) in &plan.retracted {
        println!("  RETRACT   {name}  ({reason})");
    }
    for (name, keep) in &plan.merged {
        println!("  MERGE     {name} -> {keep}");
    }
    for (name, i) in &plan.cross {
        println!("  DETECTION {name} += {instrument} {}", events[*i].peak);
    }
    for (name, i) in plan.new.iter().take(20) {
        println!("  NEW       {name}  {}", events[*i].peak);
    }
    if plan.new.len() > 20 {
        println!("  NEW       ... {} more", plan.new.len() - 20);
    }
    let held = !(plan.reinstated.is_empty() && plan.retracted.is_empty() && plan.merged.is_empty());
    match mode {
        Apply::DryRun => {
            println!("dry run: nothing written (use --apply)");
            return held;
        }
        Apply::NewOnly if held => {
            // 改动已发布编号状态的项不写：被恢复的编号连同它的探测更新一起留着
            let reinstated = std::mem::take(&mut plan.reinstated);
            plan.matched.retain(|(name, _)| !reinstated.contains(name));
            plan.retracted.clear();
            plan.merged.clear();
            println!("held for review (apply with --apply after checking): the REINSTATE/RETRACT/MERGE lines above");
        }
        _ => {}
    }

    // 单写者；中途 panic 也要放锁
    struct Lock(std::path::PathBuf);
    impl Drop for Lock {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    let lock = Lock(registry_dir.join(".lock"));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock.0)
        .unwrap_or_else(|e| panic!("registry is locked ({}): {e}", lock.0.display()));
    let now = serde_json::to_string(&Utc::now()).unwrap().trim_matches('"').to_string();
    let assigned_by = format!("{} {version}", env!("BLINK_GIT"));
    let (names, detections, history) =
        apply(&registry, &plan, &instrument, &events, version, &assigned_by, &now);
    verify_append_only(&registry.names, &names);

    let mut out = format!("{NAMES_HEADER}\n");
    for n in &names {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            n.name, n.peak_utc, n.assigned_at, n.assigned_by, n.status, n.status_reason,
            n.merged_into, n.status_changed_at
        ));
    }
    write_atomic(&registry_dir.join("names.csv"), &out);
    let mut out = format!("{DETECTIONS_HEADER}\n");
    for d in &detections {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            d.name, d.instrument, d.start, d.stop, d.peak, d.fa, d.count, d.longitude,
            d.latitude, d.altitude_km, d.version_first, d.version_last
        ));
    }
    write_atomic(&registry_dir.join("detections.csv"), &out);
    let history_path = registry_dir.join("history.csv");
    let mut out = fs::read_to_string(&history_path).unwrap_or_else(|_| format!("{HISTORY_HEADER}\n"));
    for line in &history {
        out.push_str(line);
        out.push('\n');
    }
    write_atomic(&history_path, &out);
    drop(lock);
    println!(
        "applied: names.csv {} rows, detections.csv {} rows, history.csv +{}",
        names.len(),
        detections.len(),
        history.len()
    );
    held && mode == Apply::NewOnly
}

#[cfg(test)]
mod tests {
    use super::*;

    const HXMT: &str = "Insight-HXMT/HE";
    const GBM: &str = "Fermi/GBM";

    struct Fixture {
        dir: std::path::PathBuf,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("blink_name_{tag}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join("registry")).unwrap();
            fs::write(dir.join("registry/public_instruments.txt"), format!("{HXMT}\n{GBM}\n")).unwrap();
            Self { dir }
        }

        /// 写一个目录版本：(状态, 峰值 "HH:MM:SS.ffffff") 列表，窗口为峰值 ±100 µs
        fn catalog(&self, instrument: &str, events: &[(&str, &str)], rejected: &[(&str, &str)]) -> std::path::PathBuf {
            let dir = self.dir.join("catalog");
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("manifest.json"),
                format!(
                    r#"{{"instrument": "{instrument}", "data": {{"first_day": "2026-10-20", "last_day": "2026-10-21"}}}}"#
                ),
            )
            .unwrap();
            let t = |hms: &str, d_us: i64| {
                let peak: DateTime<Utc> = format!("2026-10-21T{hms}Z").parse().unwrap();
                serde_json::to_string(&(peak + chrono::TimeDelta::microseconds(d_us)))
                    .unwrap()
                    .trim_matches('"')
                    .to_string()
            };
            let mut csv = String::from("cand_id,status,start,stop,peak,false_positive_per_year,count,longitude,latitude,altitude_km\n");
            for (status, hms) in events {
                csv.push_str(&format!(
                    "x,{status},{},{},{},1e-9,20,10.0,5.0,540.0\n",
                    t(hms, -100),
                    t(hms, 100),
                    t(hms, 0)
                ));
            }
            fs::write(dir.join("catalog.csv"), csv).unwrap();
            let mut csv = String::from("cand_id,start,reason\n");
            for (hms, reason) in rejected {
                csv.push_str(&format!("x,{},{reason}\n", t(hms, -100)));
            }
            fs::write(dir.join("rejected.csv"), csv).unwrap();
            dir
        }

        fn run(&self, catalog: &Path, version: &str) {
            cmd_name(catalog, &self.dir.join("registry"), version, Apply::All);
        }

        fn names(&self) -> Vec<NameRow> {
            load_registry(&self.dir.join("registry")).names
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn status(names: &[NameRow], name: &str) -> (String, String, String) {
        let n = names.iter().find(|n| n.name == name).unwrap();
        (n.status.clone(), n.status_reason.clone(), n.merged_into.clone())
    }

    #[test]
    fn letters_cover_the_day() {
        assert_eq!(letters(0), "AA");
        assert_eq!(letters(27), "BB");
        assert_eq!(letters(675), "ZZ");
        assert_eq!(letter_index("261021ZZ"), 675);
    }

    #[test]
    fn first_batch_is_in_time_order_and_reruns_change_nothing() {
        let f = Fixture::new("batch");
        let c = f.catalog(HXMT, &[("final", "09:00:00.0"), ("final", "03:00:00.0"), ("provisional", "12:00:00.0")], &[]);
        f.run(&c, "v1");
        let names = f.names();
        assert_eq!(names.len(), 2, "provisional events get no name");
        assert_eq!(names[0].name, "261021AA");
        assert!(names[0].peak_utc.contains("T03:00:00"));
        assert_eq!(names[1].name, "261021AB");
        // 同一版本再跑：不发新号、不改状态
        f.run(&c, "v1");
        let again = f.names();
        assert_eq!(again, names);
    }

    #[test]
    fn a_shifted_event_keeps_its_name_and_a_late_one_takes_the_next_letter() {
        let f = Fixture::new("shift");
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0")], &[]), "v1");
        // 重处理：峰值挪了 300 µs（窗口仍重叠）；补搜到一个更早的事件
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0003"), ("final", "01:00:00.0")], &[]), "v2");
        let names = f.names();
        assert_eq!(names.len(), 2);
        assert!(names[0].peak_utc.contains("T09:00:00Z"), "the published peak never changes");
        assert_eq!(names[1].name, "261021AB");
        assert!(names[1].peak_utc.contains("T01:00:00"), "letters follow assignment, not time");
        let history = fs::read_to_string(f.dir.join("registry/history.csv")).unwrap();
        assert!(history.contains("261021AA,") && history.contains(":peak,"));
    }

    #[test]
    fn a_vanished_event_is_retracted_with_its_reason_and_can_come_back() {
        let f = Fixture::new("retract");
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0"), ("final", "10:00:00.0")], &[]), "v1");
        f.run(&f.catalog(HXMT, &[("final", "10:00:00.0")], &[("09:00:00.0", "grb")]), "v2");
        assert_eq!(status(&f.names(), "261021AA"), ("retracted".into(), "rejected:grb".into(), "".into()));
        // 名字永不复用：新事件取 AC 而不是 AA
        f.run(&f.catalog(HXMT, &[("final", "10:00:00.0"), ("final", "11:00:00.0")], &[]), "v3");
        assert_eq!(f.names()[2].name, "261021AC");
        assert_eq!(status(&f.names(), "261021AA").0, "retracted");
        // 原事件重新出现：恢复原编号
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0"), ("final", "10:00:00.0"), ("final", "11:00:00.0")], &[]), "v4");
        assert_eq!(status(&f.names(), "261021AA").0, "active");
        assert_eq!(f.names().len(), 3);
    }

    #[test]
    fn new_only_holds_retractions_for_review() {
        let f = Fixture::new("newonly");
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0")], &[]), "v1");
        let c = f.catalog(HXMT, &[("final", "10:00:00.0")], &[("09:00:00.0", "grb")]);
        let held = cmd_name(&c, &f.dir.join("registry"), "v2", Apply::NewOnly);
        assert!(held);
        // 新事件照发，旧编号的撤销留着
        assert_eq!(f.names().len(), 2);
        assert_eq!(status(&f.names(), "261021AA").0, "active");
        // 人工确认后全部写入
        f.run(&c, "v2");
        assert_eq!(status(&f.names(), "261021AA").0, "retracted");
    }

    #[test]
    fn a_provisional_event_neither_retracts_nor_gets_a_name() {
        let f = Fixture::new("prov");
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0")], &[]), "v1");
        f.run(&f.catalog(HXMT, &[("provisional", "09:00:00.0")], &[]), "v2");
        assert_eq!(status(&f.names(), "261021AA").0, "active");
    }

    #[test]
    fn split_follows_the_nearest_peak_and_merge_keeps_the_oldest_name() {
        let f = Fixture::new("split");
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0")], &[]), "v1");
        // 拆分：一个旧事件 → 两个（0.0005 s 处与原峰值同窗，0.0009 s 处也重叠）
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.00005"), ("final", "09:00:00.0009")], &[]), "v2");
        let names = f.names();
        assert_eq!(names.len(), 2);
        let detections = load_registry(&f.dir.join("registry")).detections;
        let aa = detections.iter().find(|d| d.name == "261021AA").unwrap();
        assert!(aa.peak.contains("09:00:00.00005"), "the name follows the nearest peak: {}", aa.peak);
        // 合并：两个旧事件 → 一个，留先分配的 AA
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0005")], &[]), "v3");
        assert_eq!(status(&f.names(), "261021AB"), ("merged".into(), "merged".into(), "261021AA".into()));
        assert_eq!(status(&f.names(), "261021AA").0, "active");
    }

    #[test]
    fn another_instrument_adds_a_detection_under_the_same_name() {
        let f = Fixture::new("cross");
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0")], &[]), "hxmt-v1");
        // GBM 在同一位置附近 2 ms 后看到：光行时差 0 + 1 ms 放宽不够，不算同一事件
        f.run(&f.catalog(GBM, &[("final", "09:00:00.002")], &[]), "gbm-v1");
        assert_eq!(f.names().len(), 2);
        // 0.5 ms 内：同一事件，挂到 AA 下
        let f = Fixture::new("cross2");
        f.run(&f.catalog(HXMT, &[("final", "09:00:00.0")], &[]), "hxmt-v1");
        f.run(&f.catalog(GBM, &[("final", "09:00:00.0005")], &[]), "gbm-v1");
        assert_eq!(f.names().len(), 1);
        let detections = load_registry(&f.dir.join("registry")).detections;
        assert_eq!(detections.iter().filter(|d| d.name == "261021AA").count(), 2);
    }

    #[test]
    #[should_panic(expected = "not in")]
    fn a_non_public_instrument_gets_no_names() {
        let f = Fixture::new("private");
        f.run(&f.catalog("SVOM/GRM", &[("final", "09:00:00.0")], &[]), "v1");
    }

    #[test]
    #[should_panic(expected = "immutable columns")]
    fn the_published_columns_cannot_change() {
        let row = |peak: &str| NameRow {
            name: "261021AA".into(),
            peak_utc: peak.into(),
            assigned_at: "t".into(),
            assigned_by: "b".into(),
            status: "active".into(),
            status_reason: String::new(),
            merged_into: String::new(),
            status_changed_at: String::new(),
        };
        verify_append_only(&[row("a")], &[row("b")]);
    }
}
