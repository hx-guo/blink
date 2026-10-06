use crate::io::path::get_path;
use crate::types::HxmtHe;
use blink_core::error::Error;
use blink_core::types::{Attitude, MissionElapsedTime, TemporalState, Trajectory};
use chrono::prelude::*;

pub struct AttFile {
    // HDU 3: ATT_Quater
    time: Vec<f64>,
    q1: Vec<f64>,
    q2: Vec<f64>,
    q3: Vec<f64>,
}

impl AttFile {
    fn get_path(epoch: &DateTime<Utc>) -> Result<String, Error> {
        get_path(epoch, "Att")
    }

    pub fn last_modified(epoch: &DateTime<Utc>) -> Result<DateTime<Utc>, Error> {
        let path = Self::get_path(epoch)?;
        let metadata = std::fs::metadata(path)?;
        let modified_time = metadata.modified()?;
        let datetime: DateTime<Utc> = modified_time.into();
        Ok(datetime)
    }

    pub fn from_epoch(epoch: &DateTime<Utc>) -> Result<Self, Error> {
        let path = Self::get_path(epoch)?;
        Self::new(&path)
    }

    fn new(filename: &str) -> Result<Self, Error> {
        let fptr = fitsio_pure::compat::fitsfile::FitsFile::open(filename)?;

        // HDU 3: ATT_Quater
        let att = fptr.hdu("ATT_Quater")?;
        let time = att.read_col::<f64>(&fptr, "Time")?;
        let q1 = att.read_col::<f64>(&fptr, "Q1")?;
        let q2 = att.read_col::<f64>(&fptr, "Q2")?;
        let q3 = att.read_col::<f64>(&fptr, "Q3")?;

        Ok(Self { time, q1, q2, q3 })
    }
}

impl From<&AttFile> for Trajectory<MissionElapsedTime<HxmtHe>, Attitude> {
    fn from(att_file: &AttFile) -> Self {
        let points = att_file
            .time
            .iter()
            .zip(att_file.q1.iter())
            .zip(att_file.q2.iter())
            .zip(att_file.q3.iter())
            .map(|(((t, q1), q2), q3)| TemporalState {
                timestamp: MissionElapsedTime::new(*t),
                state: Attitude {
                    q1: *q1,
                    q2: *q2,
                    q3: *q3,
                },
            })
            .collect();

        Trajectory { points }
    }
}
