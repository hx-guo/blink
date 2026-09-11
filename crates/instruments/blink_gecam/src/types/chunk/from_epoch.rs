use blink_core::{error::Error, types::MissionElapsedTime};
use chrono::{TimeDelta, prelude::*};

use super::Chunk;
use crate::io::{
    CpdFile, EvtFile, PosAttFile,
    file::{find_cpd, find_grd, find_posatt},
};
use crate::types::instrument::{Gecam, Satellite};

pub(super) fn from_epoch<S: Satellite>(epoch: &DateTime<Utc>) -> Result<Chunk<S>, Error> {
    let evt_file = EvtFile::<S>::from_fits_file(find_grd::<S>(epoch)?.to_str().unwrap())?;
    // 位姿缺了整小时就没法用：没有位置就定不了位、关联不了闪电，由上层记
    // missing_data。CPD 不一样——GECAM-A 在 2025-03-12 之前一整段都没有
    // CPD 事例产品（`CPD_evt` 目录都不存在），把它当致命错误就等于把 A 星
    // 唯一落在 WWLLN 覆盖内的那 370 小时曝光整个扔掉。所以 CPD 缺失降级为
    // 「这一小时没有 CPD 数据」，GRD 照常搜，`Signal::acd` 留空。
    //
    // 放过两种「没有 CPD」：**找不到文件**，以及**文件在但是个占位空壳**
    // （零行零列的 `EVENTS` 表，实测 28,800 字节）。后者原先被算进「读坏了」
    // 那一类，代价是 **20 小时完好的 GRD 数据整小时判 corrupt**——那 20 小时
    // 的 GRD 是好文件（> 1 MB），死的只是 CPD 占位。空壳与真损坏形态上分得开，
    // 判据见 `CpdFile::from_fits_file` 里的 `is_stub`。
    //
    // 除这两种之外，文件在却读坏了仍旧报错——那是数据出了问题，不是产品线
    // 本来就没出这个东西。**GRD 的空壳照旧致命**：没有事例就没有搜索，
    // `EvtFile` 那边一条都没放松。
    let cpd_file = match find_cpd::<S>(epoch) {
        Ok(path) => CpdFile::<S>::from_fits_file(path.to_str().unwrap())?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let posatt_file = PosAttFile::from_fits_file(find_posatt::<S>(epoch)?.to_str().unwrap())?;

    Ok(Chunk {
        span: [
            MissionElapsedTime::<Gecam<S>>::from(*epoch),
            MissionElapsedTime::<Gecam<S>>::from(*epoch + TimeDelta::hours(1)),
        ],
        evt_file,
        cpd_file,
        posatt_file,
        dropped_no_ephemeris: Default::default(),
        without_attitude: Default::default(),
        dropped_single_detector: Default::default(),
        events_outside_gti: Default::default(),
        merged_gain_duplicates: Default::default(),
    })
}
