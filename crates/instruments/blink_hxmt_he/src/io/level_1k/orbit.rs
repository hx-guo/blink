use blink_core::{
    error::Error,
    types::{MissionElapsedTime, Position, TemporalState, Trajectory},
};
use chrono::prelude::*;

use crate::{io::path::get_path, types::HxmtHe};

pub struct OrbitFile {
    // HDU 1: Orbit
    time: Vec<f64>,
    lon: Vec<f64>,
    lat: Vec<f64>,
    alt: Vec<f64>,
}

impl OrbitFile {
    fn get_path(epoch: &DateTime<Utc>) -> Result<String, Error> {
        get_path(epoch, "Orbit")
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

        // HDU 1: Orbit
        let orbit = fptr.hdu("Orbit")?;
        let time = orbit.read_col::<f64>(&fptr, "Time")?;
        let lon = orbit.read_col::<f64>(&fptr, "Lon")?;
        let lat = orbit.read_col::<f64>(&fptr, "Lat")?;
        let alt = orbit.read_col::<f64>(&fptr, "Alt")?;

        Ok(Self {
            time,
            lon,
            lat,
            alt,
        })
    }
}

impl From<&OrbitFile> for Trajectory<MissionElapsedTime<HxmtHe>, Position> {
    fn from(orbit_file: &OrbitFile) -> Self {
        let points = orbit_file
            .time
            .iter()
            .zip(orbit_file.lon.iter())
            .zip(orbit_file.lat.iter())
            .zip(orbit_file.alt.iter())
            .map(|(((t, lon), lat), alt)| TemporalState {
                timestamp: MissionElapsedTime::new(*t),
                state: Position {
                    longitude: *lon,
                    latitude: *lat,
                    altitude: uom::si::f64::Length::new::<uom::si::length::meter>(*alt),
                },
            })
            .collect();

        Trajectory { points }
    }
}
