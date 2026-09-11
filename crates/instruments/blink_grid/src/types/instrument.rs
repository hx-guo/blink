use blink_core::traits::Instrument;
use chrono::prelude::*;
use std::{marker::PhantomData, str::FromStr, sync::OnceLock};

/// 天格计划的一颗星。
///
/// 每颗星 4 个 GAGG+SiPM 探测器、同一套文件格式，归档按星分目录
/// （`GRID-03B/fits7/...`），文件名带星的代号（`G03_evt_...`）。
pub trait Satellite:
    Clone + Copy + PartialEq + Eq + PartialOrd + Ord + std::fmt::Debug + Send + Sync + 'static
{
    /// 归档里的目录名，也是输出目录名
    const DIR: &'static str;
    /// 文件名前缀
    const CODE: &'static str;
    /// 四路探测器是否共用一条读出链（共帧读出）。
    ///
    /// 实测判据是跨探头相邻间隔的分布形态：共帧星的跨探头间隔要么恰好 0、要么
    /// ≥ 帧长，中间是空的——GRID-02 落在开区间 (0, τ) 的只有 4.8e-6（四路独立模型
    /// 预期 1.9e-2，低 3900 倍），GRID-04 是 4.3e-4（预期 5.5e-2，低 130 倍）；
    /// GRID-03B 则连续铺满（34.4%，与独立模型符合）。共帧的机制是：任一路击中就开
    /// 一个帧，帧内每路最多贡献 1 个事例，**全帧共用触发那一击的时戳**，帧结束才能
    /// 开下一帧。
    ///
    /// 这个常量只有一个用途：**共帧星上同戳是读出结构的必然产物，不是物理**，所以
    /// 带电粒子的同戳判据（见 `chunk::search`）对它们无意义，要整个跳过。见
    /// `OPEN-QUESTIONS.md` 未决项 17。
    const SHARED_FRAME: bool;
    /// 事例产品（`fits7`）的第一天。不是发射日：盲搜从有数据的那天起算。
    fn first_day() -> NaiveDate;
}

macro_rules! satellite {
    ($ty:ident, $dir:literal, $code:literal, $frame:literal,
     ($y:literal, $m:literal, $d:literal), $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
        pub struct $ty;
        impl Satellite for $ty {
            const DIR: &'static str = $dir;
            const CODE: &'static str = $code;
            const SHARED_FRAME: bool = $frame;
            fn first_day() -> NaiveDate {
                NaiveDate::from_ymd_opt($y, $m, $d).unwrap()
            }
        }
    };
}

satellite!(
    Sat02,
    "GRID-02",
    "G02",
    true,
    (2020, 11, 14),
    "GRID-02：事例产品 2020-11-14 .. 2021-03-08，75 天"
);
satellite!(
    Sat03B,
    "GRID-03B",
    "G03",
    false,
    (2022, 3, 11),
    "GRID-03B：事例产品 2022-03-11 .. 2024-08-01，789 天"
);
satellite!(
    Sat04,
    "GRID-04",
    "G04",
    true,
    (2022, 3, 11),
    "GRID-04：事例产品 2022-03-11 .. 2024-08-22，527 天（2023-03..12 缺）"
);
satellite!(
    Sat07,
    "GRID-07",
    "G07",
    true,
    (2024, 1, 1),
    "GRID-07：事例产品 2024-01-01 .. 2024-07-22，195 天。\
     共帧已实测确认：跨探头间隔落在 (0, τ) 只有 1.45e-2 倍于四路独立模型的预期（低 69 倍），\
     同戳簇长分布（长度 2 有 99333 段、3 有 1354 段、4 有 228 段）远超该速率下的偶然值 5.7e-4。\
     与 02/04 不同的是它的帧原点不锚定绝对时钟（时戳对 τ 取模是均匀的，变异系数 0.0027）——\
     帧周期本身在 120/121/122 tick 之间抖，几千帧下来相位就随机游走开了。"
);

/// 某颗天格星作为一台仪器。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Grid<S: Satellite>(PhantomData<S>);

impl<S: Satellite> Instrument for Grid<S> {
    type Chunk = crate::types::Chunk<S>;

    /// 事例表 `TIME` 的零点。文件头 `DATE_REF = '2018-01-01T00:00:00.000'`；
    /// 对账：`TIME = 132217726.086` ↔ `DATE_OBS = '2022-03-11T07:08:46.083'`，
    /// 按 UTC 直接数 1530 天 + 25726 s = 132217726 s，完全一致。2017 年以后
    /// 没有新闰秒，`MissionElapsedTime` 的闰秒表在这里恒为零修正。
    fn ref_time() -> &'static DateTime<Utc> {
        static REF_TIME: OnceLock<DateTime<Utc>> = OnceLock::new();
        REF_TIME
            .get_or_init(|| DateTime::<Utc>::from_str("2018-01-01T00:00:00.000000000 UTC").unwrap())
    }

    fn launch_day() -> NaiveDate {
        S::first_day()
    }

    fn name() -> &'static str {
        S::DIR
    }
}

pub type Grid02 = Grid<Sat02>;
pub type Grid03B = Grid<Sat03B>;
pub type Grid04 = Grid<Sat04>;
pub type Grid07 = Grid<Sat07>;
