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
    // CPD 和位姿缺了整小时就没法用：没有位置就定不了位、关联不了闪电，
    // 没有 CPD 就少了唯一一条直接的带电粒子判据。缺哪个都当这一小时不可用，
    // 由上层记 missing_data，而不是悄悄降级。
    let cpd_file = CpdFile::<S>::from_fits_file(find_cpd::<S>(epoch)?.to_str().unwrap())?;
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
    })
}
