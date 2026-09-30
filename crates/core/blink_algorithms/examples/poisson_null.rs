//! 纯泊松零假设检验：在恒定计数率的合成事例流上跑 HXMT 同款搜索，
//! 输出每个候选的 `fa`，用来看论文 Fig. 6 左侧那支陡峭幂律是否来自泊松涨落。
//!
//! 用法：`cargo run --release -p blink_algorithms --example poisson_null -- <rate_hz> <hours> <seed>`
//! 输出（stdout）：CSV `hour,start,count,mean,duration_us,fa`；stderr 打印每小时进度。
//!
//! 按小时分块，与真实搜索一样每块 `gti = [[0, 3600]]`；各块用 `seed + hour` 播种，
//! 所以可以按小时切开并行跑，结果与串行逐位相同。

use blink_algorithms::snapshot_stepping::{SearchConfig, search_new};
use blink_core::{
    error::Error,
    traits::{Chunk, Event, Instrument},
    types::{Coverage, MissionElapsedTime, Signal},
};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::Serialize;
use std::sync::OnceLock;
use uom::si::f64::Time;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct NullInstrument;

struct NullChunk;

impl Chunk for NullChunk {
    type Event = NullEvent;

    fn from_epoch(_: &DateTime<Utc>) -> Result<Self, Error> {
        Err(Error::Unknown)
    }
    fn search(&self) -> Vec<Signal<Self::Event>> {
        Vec::new()
    }
    fn last_modified(_: &DateTime<Utc>) -> Result<DateTime<Utc>, Error> {
        Err(Error::Unknown)
    }
    fn coverage(&self) -> Coverage {
        Coverage {
            span_seconds: 0.0,
            masked_seconds: 0.0,
        }
    }
}

impl Instrument for NullInstrument {
    type Chunk = NullChunk;

    fn ref_time() -> &'static DateTime<Utc> {
        static REF_TIME: OnceLock<DateTime<Utc>> = OnceLock::new();
        REF_TIME.get_or_init(|| Utc.with_ymd_and_hms(2012, 1, 1, 0, 0, 0).unwrap())
    }
    fn launch_day() -> NaiveDate {
        NaiveDate::from_ymd_opt(2017, 6, 15).unwrap()
    }
    fn name() -> &'static str {
        "poisson-null"
    }
}

#[derive(Serialize, Debug, Clone)]
struct NullEvent {
    seconds: f64,
}

impl Event for NullEvent {
    type Instrument = NullInstrument;
    type ChannelType = u16;

    fn time(&self) -> MissionElapsedTime<NullInstrument> {
        MissionElapsedTime::new(self.seconds)
    }
    fn channel(&self) -> u16 {
        100
    }
    fn group(&self) -> u8 {
        0
    }
    fn keep(&self) -> bool {
        true
    }
}

/// splitmix64：种子扩散；xoshiro256** 的标准播种方式。
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256**：工作区里没有 rand，这里自带一个，足够做泊松过程。
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        Rng([
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ])
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// (0, 1] 上的均匀数，取 53 位；排除 0 以免 ln(0)。
    fn uniform_open0(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

const CHUNK_SECONDS: f64 = 3600.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: poisson_null <rate_hz> <hours> <seed>");
        std::process::exit(2);
    }
    let rate: f64 = args[1].parse().expect("rate_hz");
    let hours: u64 = args[2].parse().expect("hours");
    let seed: u64 = args[3].parse().expect("seed");

    println!("hour,start,count,mean,duration_us,fa");
    for hour in 0..hours {
        let mut rng = Rng::new(seed.wrapping_add(hour));
        // 指数间隔生成齐次泊松过程
        let mut events = Vec::with_capacity((rate * CHUNK_SECONDS * 1.01) as usize);
        let mut t = -rng.uniform_open0().ln() / rate;
        while t < CHUNK_SECONDS {
            events.push(NullEvent { seconds: t });
            t += -rng.uniform_open0().ln() / rate;
        }

        let start = MissionElapsedTime::<NullInstrument>::new(0.0);
        let stop = MissionElapsedTime::<NullInstrument>::new(CHUNK_SECONDS);
        // 与 blink_hxmt_he 的 chunk/search.rs 相同的配置
        let candidates = search_new(
            &events,
            1,
            start,
            stop,
            &[[start, stop]],
            SearchConfig {
                min_duration: Time::new::<uom::si::time::microsecond>(0.0),
                max_duration: Time::new::<uom::si::time::millisecond>(1.0),
                neighbor: Time::new::<uom::si::time::second>(1.0),
                hollow: Time::new::<uom::si::time::millisecond>(10.0),
                false_positive_per_year: 20.0,
                min_number: 8,
                coincidence: 1,
            },
        );

        for c in &candidates {
            println!(
                "{},{:.9},{},{:.6},{:.3},{:e}",
                hour,
                (c.start - start).get::<uom::si::time::second>(),
                c.count,
                c.mean,
                c.bin_size_best.get::<uom::si::time::microsecond>(),
                c.false_positive_per_year()
            );
        }
        eprintln!(
            "hour {}: {} events, {} candidates",
            hour,
            events.len(),
            candidates.len()
        );
    }
}
