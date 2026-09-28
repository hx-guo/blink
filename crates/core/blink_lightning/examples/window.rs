//! 打印某一时刻前后若干秒内 WWLLN 记录的全部闪电（CSV：time,lat,lon）。
//!
//! 用法：cargo run -p blink_lightning --example window -- 2022-10-04T00:09:56.543 60
//! 数据库路径取 WWLLN_DB_PATH，与搜索相同。
use blink_lightning::database::get_lightnings;
use chrono::{Duration, NaiveDateTime};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let centre = NaiveDateTime::parse_from_str(&args[1], "%Y-%m-%dT%H:%M:%S%.f")
        .expect("time like 2022-10-04T00:09:56.543")
        .and_utc();
    let half: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(60.0);
    let d = Duration::microseconds((half * 1e6) as i64);
    println!("time,lat,lon");
    for l in get_lightnings(centre - d, centre + d) {
        println!("{},{},{}", l.time.format("%Y-%m-%dT%H:%M:%S%.6f"), l.lat, l.lon);
    }
}
