use crate::error::Error;
use crate::traits::Event;
use crate::types::{Coverage, ExclusionReason, ExposureSample, Signal};
use chrono::prelude::*;

pub trait Chunk {
    type Event: Event;

    fn from_epoch(epoch: &DateTime<Utc>) -> Result<Self, Error>
    where
        Self: Sized;
    fn search(&self) -> Vec<Signal<Self::Event>>;
    fn last_modified(epoch: &DateTime<Utc>) -> Result<DateTime<Utc>, Error>;

    /// 数据载入成功之后、搜索之前的体检。
    ///
    /// 返回 `Some(reason)` 表示这个单元不该被搜索（搜了也是错的），调用方必须
    /// 把它记成 excluded，而不是照常搜索、静默产出一堆假候选或空结果。
    /// 默认不排除任何东西 —— 仪器按自己的失效模式覆写。
    fn exclusion(&self) -> Option<ExclusionReason> {
        None
    }

    /// 本单元的曝光核算（率估计的分母）。
    fn coverage(&self) -> Coverage;

    /// 体检用到的实测量，`(名字, 数值)`。无论 [`Self::exclusion`] 判没判，
    /// 都随每个单元记录下来，这样阈值本身事后还能审计。
    fn diagnostics(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }

    /// 曝光统计（管线 S5）：在**搜索实际用到的活时间**里每 `step_seconds` 取一点，
    /// 给出星下点。口径必须与搜索一致——搜索扣掉的（数据缺口、配置不全、饱和……）
    /// 这里也扣，否则率的分子分母不配。只对通过 [`Self::exclusion`] 的单元调用。
    /// 默认 `None`：仪器还没实现，不出曝光（而不是出一份口径不对的曝光）。
    fn exposure_samples(&self, _step_seconds: f64) -> Option<Vec<ExposureSample>> {
        None
    }
}
