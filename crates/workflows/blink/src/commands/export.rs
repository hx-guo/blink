//! 网页数据导出（管线 S6）：目录 + 登记表 + vet 光变曲线 → 静态站读的 JSON。
//!
//! 网页不另算任何数，只读这里写出的文件（docs/tgf-pipeline-design.md 第 7 节）：
//!
//! ```text
//! <out>/index.json                各仪器的版本、条数、数据范围、活时间、是否公开
//! <out>/latest.json               最近的事件（首页）
//! <out>/names.json                编号 → 状态与各仪器的探测（含已撤销/合并）
//! <out>/names/<yymmddAA>.json     同上，单个编号（编号页只取这一个小文件）
//! <out>/<slug>/catalog.json       列式：每列一个数组
//! <out>/<slug>/catalog.csv        下载用，比 blink catalog 的多一列 name
//! <out>/<slug>/manifest.json      原样拷贝
//! <out>/<slug>/exposure.json      曝光（S5）：5° 经纬格、1° 纬度、半小时视太阳时的活时间（秒）
//! <out>/<slug>/events/<cand_id>.json  单事件：全部字段 + 光变曲线
//! ```
//!
//! `--public-only` 只导出登记表 `public_instruments.txt` 里的仪器（公开站）；不加则
//! 全部导出、未公开的仪器在 index 里标 `public: false`（内部站）。
//!
//! 光变曲线来自 `<data>/<slug>/YYYY/MM/YYYYMMDD_vet.json`；没复核过的事件只有元数据。

use crate::commands::vet::Vet;
use chrono::{DateTime, Utc};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

/// catalog.json 里转成数值的列，其余按字符串
const NUMERIC: &[&str] = &[
    "duration_us",
    "count",
    "false_positive_per_year",
    "longitude",
    "latitude",
    "altitude_km",
    "neighbors_10min",
    "grb_z",
    "grb_max_box_share",
    "electron_beam",
    "beam_p",
    "det_share_max",
    "n",
    "n_acd",
    "n_acd_multi",
    "n_bg",
    "n_acd_bg",
];

fn read_csv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let content =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut lines = content.lines();
    let header = lines
        .next()
        .map(|h| h.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let rows = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect();
    (header, rows)
}

fn cell(value: &str, numeric: bool) -> Value {
    if !numeric {
        return if value.is_empty() { Value::Null } else { Value::from(value) };
    }
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .map(Value::from)
        .unwrap_or(Value::Null)
}

fn write_json(path: &Path, value: &Value) {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).expect("failed to create output directory");
    }
    fs::write(path, serde_json::to_string(value).expect("serialize json"))
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

/// 登记表：(仪器, start) → 编号，以及 names.csv 全表
struct Names {
    by_detection: HashMap<(String, String), String>,
    rows: Vec<BTreeMap<String, String>>,
    detections: Vec<BTreeMap<String, String>>,
    public: BTreeSet<String>,
}

fn read_names(registry: &Path) -> Names {
    let table = |file: &str| -> Vec<BTreeMap<String, String>> {
        let path = registry.join(file);
        if !path.exists() {
            return Vec::new();
        }
        let (header, rows) = read_csv(&path);
        rows.into_iter()
            .map(|r| header.iter().cloned().zip(r).collect())
            .collect()
    };
    let rows = table("names.csv");
    let detections = table("detections.csv");
    let by_detection = detections
        .iter()
        .map(|d| ((d["instrument"].clone(), d["start"].clone()), d["name"].clone()))
        .collect();
    let public = fs::read_to_string(registry.join("public_instruments.txt"))
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    Names {
        by_detection,
        rows,
        detections,
        public,
    }
}

/// 一天的 vet 记录，按 start 索引
fn read_vets(data: &Path, slug: &str, day: &str) -> HashMap<DateTime<Utc>, Vet> {
    let path = data
        .join(slug)
        .join(&day[..4])
        .join(&day[4..6])
        .join(format!("{day}_vet.json"));
    let Ok(content) = fs::read_to_string(&path) else {
        return HashMap::new();
    };
    let vets: Vec<Vet> = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
    vets.into_iter().map(|v| (v.start, v)).collect()
}

fn light_curve_json(vet: &Vet) -> Value {
    match &vet.light_curve {
        Some(lc) => json!({
            "peak": lc.peak,
            // [t_us, 探测器, 道址, ACD 位]：逐事例对象太占地方
            "columns": ["t_us", "det", "channel", "acd"],
            "events": lc.events.iter()
                .map(|e| json!([(e.t_us * 100.0).round() / 100.0, e.det, e.channel, e.acd]))
                .collect::<Vec<_>>(),
            "context_bin_s": lc.context_bin_s,
            "context_start_s": lc.context_start_s,
            "context": lc.context,
        }),
        None => Value::Null,
    }
}

/// 网页用的曝光：目录的 1° 地图并成 5° 格，另给纬度与视太阳时的边缘分布
const WEB_CELL_DEG: u16 = 5;

fn coarse_exposure(path: &Path) -> Option<Value> {
    let full: Value = serde_json::from_str(&fs::read_to_string(path).ok()?).expect("bad exposure.json");
    let cells = |kind: &str| -> Vec<(u16, u16, f64)> {
        full[kind]["cells"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|c| (c[0].as_u64().unwrap() as u16, c[1].as_u64().unwrap() as u16, c[2].as_f64().unwrap()))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut map: BTreeMap<(u16, u16), f64> = BTreeMap::new();
    let mut latitude = vec![0.0f64; 180];
    for (i, j, s) in cells("map") {
        *map.entry((i / WEB_CELL_DEG, j / WEB_CELL_DEG)).or_default() += s;
        latitude[i as usize] += s;
    }
    let mut lst = vec![0.0f64; 48];
    for (_, j, s) in cells("lst") {
        lst[j as usize] += s;
    }
    let round = |s: f64| (s * 10.0).round() / 10.0;
    Some(json!({
        "live_seconds": full["live_seconds"],
        "no_ephemeris_seconds": full["no_ephemeris_seconds"],
        "days_with_exposure": full["days_with_exposure"],
        "days": full["days"],
        "demo": full.get("demo").cloned().unwrap_or(Value::Bool(false)),
        "cell_deg": WEB_CELL_DEG,
        // [纬度下沿, 经度下沿（0..360）, 秒]
        "map": map.into_iter()
            .map(|((i, j), s)| json!([i as i32 * WEB_CELL_DEG as i32 - 90, j * WEB_CELL_DEG, round(s)]))
            .collect::<Vec<_>>(),
        // 下标 = ⌊lat + 90⌋
        "latitude": latitude.into_iter().map(round).collect::<Vec<_>>(),
        // 下标 = 半小时
        "lst": lst.into_iter().map(round).collect::<Vec<_>>(),
    }))
}

pub struct ExportArgs<'a> {
    pub catalogs: &'a [PathBuf],
    pub registry: &'a Path,
    pub data: &'a Path,
    pub out: &'a Path,
    pub public_only: bool,
    pub latest: usize,
}

pub fn cmd_export(args: &ExportArgs) {
    let names = read_names(args.registry);
    let mut instruments = Vec::new();
    let mut latest: Vec<Value> = Vec::new();
    // 编号 → 各仪器探测对应的 cand_id，供编号页找事件文件
    let mut cand_of: HashMap<(String, String), String> = HashMap::new();
    let mut exported: BTreeSet<String> = BTreeSet::new();

    for dir in args.catalogs {
        let manifest: Value = serde_json::from_str(
            &fs::read_to_string(dir.join("manifest.json"))
                .unwrap_or_else(|e| panic!("{}: no manifest.json: {e}", dir.display())),
        )
        .expect("bad manifest.json");
        let instrument = manifest["instrument"].as_str().expect("manifest.instrument").to_string();
        let slug = manifest["slug"].as_str().expect("manifest.slug").to_string();
        let public = names.public.contains(&instrument);
        if args.public_only && !public {
            eprintln!("export: skip {instrument} (not public)");
            continue;
        }
        exported.insert(instrument.clone());
        let out = args.out.join(&slug);
        let (header, rows) = read_csv(&dir.join("catalog.csv"));
        let col = |name: &str| {
            header
                .iter()
                .position(|h| h == name)
                .unwrap_or_else(|| panic!("catalog.csv has no `{name}` column"))
        };
        let (i_id, i_start, i_status) = (col("cand_id"), col("start"), col("status"));
        let name_of = |row: &[String]| {
            names
                .by_detection
                .get(&(instrument.clone(), row[i_start].clone()))
                .cloned()
        };

        // 列式 catalog.json，多一列 name
        let mut columns: Map<String, Value> = Map::new();
        columns.insert(
            "name".into(),
            rows.iter().map(|r| name_of(r).map(Value::from).unwrap_or(Value::Null)).collect(),
        );
        for (j, h) in header.iter().enumerate() {
            if h == "instrument" {
                continue;
            }
            let numeric = NUMERIC.contains(&h.as_str());
            columns.insert(h.clone(), rows.iter().map(|r| cell(&r[j], numeric)).collect());
        }
        write_json(
            &out.join("catalog.json"),
            &json!({ "instrument": instrument, "slug": slug, "n": rows.len(), "columns": columns }),
        );

        let mut csv = format!("name,{}\n", header.join(","));
        for r in &rows {
            csv.push_str(&format!("{},{}\n", name_of(r).unwrap_or_default(), r.join(",")));
        }
        fs::write(out.join("catalog.csv"), csv).expect("failed to write catalog.csv");
        write_json(&out.join("manifest.json"), &manifest);
        if let Some(exposure) = coarse_exposure(&dir.join("exposure.json")) {
            write_json(&out.join("exposure.json"), &exposure);
        }

        // 单事件文件：按天读 vet
        let mut by_day: BTreeMap<String, Vec<&Vec<String>>> = BTreeMap::new();
        for r in &rows {
            let day: String = r[i_start].chars().filter(char::is_ascii_digit).take(8).collect();
            by_day.entry(day).or_default().push(r);
        }
        let mut n_lc = 0usize;
        for (day, day_rows) in by_day {
            let vets = read_vets(args.data, &slug, &day);
            for r in day_rows {
                let mut fields: Map<String, Value> = Map::new();
                fields.insert("instrument".into(), Value::from(instrument.clone()));
                fields.insert("slug".into(), Value::from(slug.clone()));
                fields.insert("name".into(), name_of(r).map(Value::from).unwrap_or(Value::Null));
                for (j, h) in header.iter().enumerate() {
                    if h != "instrument" {
                        fields.insert(h.clone(), cell(&r[j], NUMERIC.contains(&h.as_str())));
                    }
                }
                let start: Option<DateTime<Utc>> = r[i_start].parse().ok();
                let lc = start
                    .and_then(|s| vets.get(&s))
                    .map(light_curve_json)
                    .unwrap_or(Value::Null);
                n_lc += !lc.is_null() as usize;
                fields.insert("light_curve".into(), lc);
                write_json(&out.join("events").join(format!("{}.json", r[i_id])), &Value::Object(fields));
                if let Some(name) = name_of(r) {
                    cand_of.insert((name, instrument.clone()), r[i_id].clone());
                }
            }
        }

        // 最近事件候选
        let pick = |name: &str| header.iter().position(|h| h == name);
        for r in &rows {
            let get = |c: &str| pick(c).map(|j| r[j].clone()).unwrap_or_default();
            latest.push(json!({
                "name": name_of(r),
                "cand_id": r[i_id],
                "instrument": instrument,
                "slug": slug,
                "status": r[i_status],
                "peak": get("peak"),
                "longitude": cell(&get("longitude"), true),
                "latitude": cell(&get("latitude"), true),
                "false_positive_per_year": cell(&get("false_positive_per_year"), true),
                "count": cell(&get("count"), true),
                "duration_us": cell(&get("duration_us"), true),
            }));
        }

        let counts = &manifest["counts"];
        let n_final = rows.iter().filter(|r| r[i_status] == "final").count();
        instruments.push(json!({
            "instrument": instrument,
            "slug": slug,
            "public": public,
            "n_events": rows.len(),
            "n_final": n_final,
            "n_named": rows.iter().filter(|r| name_of(r).is_some()).count(),
            "electron_beams": counts["electron_beams"],
            "first_day": manifest["data"]["first_day"],
            "last_day": manifest["data"]["last_day"],
            "searched_seconds": manifest["data"]["searched_seconds"],
            "generated_at": manifest["generated_at"],
            "blink_git": manifest["blink_git"],
            // 演示数据（web/scripts/make_demo_data.py）带这个标记，网页据此挂横幅
            "demo": manifest.get("demo").cloned().unwrap_or(Value::Bool(false)),
        }));
        eprintln!(
            "export {instrument}: {} events ({n_final} final), {n_lc} with light curves -> {}",
            rows.len(),
            out.display()
        );
    }

    // 编号表：只含已导出仪器的探测；编号本身（含撤销/合并）全部保留
    let mut names_json: Vec<Value> = Vec::new();
    for n in &names.rows {
        let detections: Vec<Value> = names
            .detections
            .iter()
            .filter(|d| d["name"] == n["name"] && exported.contains(&d["instrument"]))
            .map(|d| {
                json!({
                    "instrument": d["instrument"],
                    "slug": d["instrument"].replace('/', "_"),
                    "cand_id": cand_of.get(&(n["name"].clone(), d["instrument"].clone())),
                    "start": d["start"],
                    "peak": d["peak"],
                    "longitude": cell(&d["longitude"], true),
                    "latitude": cell(&d["latitude"], true),
                })
            })
            .collect();
        if detections.is_empty() && n["status"] == "active" {
            continue;
        }
        names_json.push(json!({
            "name": n["name"],
            "peak_utc": n["peak_utc"],
            "assigned_at": n["assigned_at"],
            "status": n["status"],
            "status_reason": n["status_reason"],
            "merged_into": n["merged_into"],
            "detections": detections,
        }));
    }
    for entry in &names_json {
        let name = entry["name"].as_str().expect("name");
        write_json(&args.out.join("names").join(format!("{name}.json")), entry);
    }
    write_json(&args.out.join("names.json"), &Value::from(names_json.clone()));

    latest.sort_by(|a, b| b["peak"].as_str().cmp(&a["peak"].as_str()));
    latest.truncate(args.latest);
    write_json(&args.out.join("latest.json"), &Value::from(latest));

    write_json(
        &args.out.join("index.json"),
        &json!({
            "generated_at": Utc::now(),
            "public_only": args.public_only,
            "n_names": names_json.len(),
            "instruments": instruments,
        }),
    );
    eprintln!(
        "export: {} instruments, {} names -> {}",
        instruments.len(),
        names_json.len(),
        args.out.display()
    );
}
