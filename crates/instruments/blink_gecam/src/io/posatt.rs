/*
gb_posatt_220615_00_v00.fits
No.    Name             Type          Dimensions
  0  PrimaryHDU     PrimaryHDU        ()
  1  Orbit_Attitude BinTableHDU       3700R x 22C
     TIME, Q1..Q4, wx/wy/wz, Orb_Type,
     X/Y/Z_J2000, VX/VY/VZ_J2000, X/Y/Z_WGS84, VX/VY/VZ_WGS84, TIME_Quality
*/

//! 位姿表：位置与姿态在同一张表里，1 Hz 采样，一小时 3700 点（越过整点约
//! 100 s，与事例文件同样有重叠）。位置只给地固系直角坐标（米），要自己折成
//! 经纬高，见 [`crate::io::geodetic`]。

use blink_core::types::{Attitude, MissionElapsedTime, Position, TemporalState, Trajectory};
use uom::si::f64::Length;
use uom::si::length::meter;

use crate::io::geodetic::wgs84_to_geodetic;
use crate::types::instrument::{Gecam, Satellite};

pub struct PosAttFile {
    time: Vec<f64>,
    q1: Vec<f32>,
    q2: Vec<f32>,
    q3: Vec<f32>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

impl PosAttFile {
    pub fn from_fits_file(path: &str) -> Result<Self, fitsio::errors::Error> {
        let mut fptr = fitsio::FitsFile::open(path)?;
        let hdu = fptr.hdu("Orbit_Attitude")?;

        Ok(Self {
            time: hdu.read_col::<f64>(&mut fptr, "TIME")?,
            // 表里有 Q1..Q4，`Attitude` 只收三个分量；标量分量放在 Q1 还是 Q4
            // 尚未核实，见本 crate 的 OPEN-QUESTIONS。姿态目前只作元数据随候选
            // 记录，不参与任何判据，所以先按表里的顺序原样取前三个。
            q1: hdu.read_col::<f32>(&mut fptr, "Q1")?,
            q2: hdu.read_col::<f32>(&mut fptr, "Q2")?,
            q3: hdu.read_col::<f32>(&mut fptr, "Q3")?,
            x: hdu.read_col::<f64>(&mut fptr, "X_WGS84")?,
            y: hdu.read_col::<f64>(&mut fptr, "Y_WGS84")?,
            z: hdu.read_col::<f64>(&mut fptr, "Z_WGS84")?,
        })
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

impl<S: Satellite> From<&PosAttFile> for Trajectory<MissionElapsedTime<Gecam<S>>, Position> {
    fn from(file: &PosAttFile) -> Self {
        let points = file
            .time
            .iter()
            .zip(file.x.iter())
            .zip(file.y.iter())
            .zip(file.z.iter())
            .map(|(((time, x), y), z)| {
                let geodetic = wgs84_to_geodetic(*x, *y, *z);
                TemporalState {
                    timestamp: MissionElapsedTime::new(*time),
                    state: Position {
                        longitude: geodetic.longitude_deg,
                        latitude: geodetic.latitude_deg,
                        altitude: Length::new::<meter>(geodetic.altitude_m),
                    },
                }
            })
            .collect();

        Trajectory { points }
    }
}

impl<S: Satellite> From<&PosAttFile> for Trajectory<MissionElapsedTime<Gecam<S>>, Attitude> {
    fn from(file: &PosAttFile) -> Self {
        let points = file
            .time
            .iter()
            .zip(file.q1.iter())
            .zip(file.q2.iter())
            .zip(file.q3.iter())
            .map(|(((time, q1), q2), q3)| TemporalState {
                timestamp: MissionElapsedTime::new(*time),
                state: Attitude {
                    q1: *q1 as f64,
                    q2: *q2 as f64,
                    q3: *q3 as f64,
                },
            })
            .collect();

        Trajectory { points }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::instrument::SatB;
    use uom::si::length::kilometer;

    /// 赤道面上正东 90°、|r| = 6.97e6 m 的一个采样点，与 GECAM-B 实测半径同量级。
    fn one_point() -> PosAttFile {
        PosAttFile {
            time: vec![100.0],
            q1: vec![0.0],
            q2: vec![0.0],
            q3: vec![0.0],
            x: vec![0.0],
            y: vec![6.97e6],
            z: vec![0.0],
        }
    }

    #[test]
    fn cartesian_metres_become_degrees_and_a_height_above_the_ellipsoid() {
        let trajectory: Trajectory<MissionElapsedTime<Gecam<SatB>>, Position> =
            (&one_point()).into();
        let position = &trajectory.points[0].state;
        assert!((position.longitude - 90.0).abs() < 1e-9);
        assert!(position.latitude.abs() < 1e-9);
        // 6.97e6 m 减去长半轴 6378137 m ≈ 592 km，不是 6970 km 的地心距
        let altitude_km = position.altitude.get::<kilometer>();
        assert!((altitude_km - 591.863).abs() < 0.01, "got {altitude_km} km");
    }
}
