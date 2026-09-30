//! 逐小时统计搜索实际用到的事例率（`keep()` 之后的 CsI 事例），给泊松零假设模拟加权用。
//!
//! 用法：从 stdin 逐行读 `YYYY-MM-DDTHH`，stdout 输出 CSV
//! `epoch,n_kept,live_seconds`；`live_seconds` 与搜索同一口径（事例流推出的 GTI 之和）。
//! 读不到的小时打到 stderr 并跳过。

use blink_core::traits::{Chunk as _, Event as _};
use blink_hxmt_he::algorithms::live_time;
use blink_hxmt_he::types::chunk::Chunk;
use chrono::{NaiveDateTime, TimeZone, Utc};
use std::io::BufRead;

fn main() {
    println!("epoch,n_kept,live_seconds");
    for line in std::io::stdin().lock().lines() {
        let epoch = line.expect("stdin");
        let epoch = epoch.trim();
        if epoch.is_empty() {
            continue;
        }
        let Ok(naive) = NaiveDateTime::parse_from_str(&format!("{epoch}:00:00"), "%Y-%m-%dT%H:%M:%S")
        else {
            eprintln!("{epoch}: bad epoch");
            continue;
        };
        let chunk = match Chunk::from_epoch(&Utc.from_utc_datetime(&naive)) {
            Ok(chunk) => chunk,
            Err(e) => {
                eprintln!("{epoch}: {e:?}");
                continue;
            }
        };
        let events = chunk
            .event_file
            .into_iter()
            .filter(|event| event.keep())
            .collect::<Vec<_>>();
        let gti = live_time::gti_from_events(
            &events,
            [chunk.span[0], chunk.span[1]],
            live_time::DEFAULT_FALSE_GAP_BUDGET,
        );
        let live: f64 = gti.iter().map(|seg| seg[1].met() - seg[0].met()).sum();
        println!("{epoch},{},{live:.3}", events.len());
    }
}
