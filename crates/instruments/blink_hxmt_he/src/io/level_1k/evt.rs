use super::super::path::get_path;
use crate::types::{Detector, Event, Scintillator};
use blink_core::{error::Error, types::MissionElapsedTime};
use chrono::prelude::*;

pub struct EventFile {
    // HDU 1: Events
    time: Vec<f64>,
    det_id: Vec<u8>,
    channel: Vec<u8>,
    pulse_width: Vec<u8>,
    acd: Vec<[bool; 18]>,
    event_type: Vec<u8>,
    // flag: Vec<u8>,
}

impl EventFile {
    fn get_path(epoch: &DateTime<Utc>) -> Result<String, Error> {
        get_path(epoch, "Evt")
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

    pub fn times(&self) -> &[f64] {
        &self.time
    }

    pub fn det_ids(&self) -> &[u8] {
        &self.det_id
    }

    pub fn channels(&self) -> &[u8] {
        &self.channel
    }

    pub fn pulse_widths(&self) -> &[u8] {
        &self.pulse_width
    }

    pub fn event_types(&self) -> &[u8] {
        &self.event_type
    }

    fn new(filename: &str) -> Result<Self, Error> {
        let mut fptr = fitsio::FitsFile::open(filename)?;

        // HDU 1: Events
        let events = fptr.hdu("Events")?;
        let time = events.read_col::<f64>(&mut fptr, "Time")?;
        let det_id = events.read_col::<u8>(&mut fptr, "Det_ID")?;
        let channel = events.read_col::<u8>(&mut fptr, "Channel")?;
        let pulse_width = events.read_col::<u8>(&mut fptr, "Pulse_Width")?;

        let acd = read_acd(&mut fptr)?;

        let event_type = events.read_col::<u8>(&mut fptr, "Event_Type")?;
        // let flag = events.read_col::<u8>(&mut fptr, "FLAG")?;

        Ok(Self {
            time,
            det_id,
            channel,
            pulse_width,
            acd,
            event_type,
            // flag,
        })
    }
}

impl<'a> IntoIterator for &'a EventFile {
    type Item = Event;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            event_file: self,
            index: 0,
        }
    }
}

pub struct Iter<'a> {
    event_file: &'a EventFile,
    index: usize,
}

impl Iterator for Iter<'_> {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.event_file.time.len() {
            let event = Event::new(
                MissionElapsedTime::new(self.event_file.time[self.index]),
                self.event_file.channel[self.index],
                Detector {
                    id: self.event_file.det_id[self.index],
                    scintillator: if self.event_file.pulse_width[self.index] < 75 {
                        Scintillator::Nai
                    } else {
                        Scintillator::Csi
                    },
                },
                self.event_file.event_type[self.index] == 1,
                self.event_file.acd[self.index],
            );
            self.index += 1;
            Some(event)
        } else {
            None
        }
    }
}

/// 读 `ACD` 列（FITS `18X`，每行 3 字节，位序高位在前，第 j 位是第 j 块 ACD）。
///
/// 不能用 `read_col`：fitsio 对位列每行只取一个元素，cfitsio 却按字节跨行连续填，
/// 结果第 i 个事例拿到的是整列第 i 个字节（约第 i/3 行），且只有 8 位。
fn read_acd(fptr: &mut fitsio::FitsFile) -> Result<Vec<[bool; 18]>, Error> {
    let hdu = fptr.hdu("Events")?;
    let fitsio::hdu::HduInfo::TableInfo {
        column_descriptions,
        num_rows,
    } = &hdu.info
    else {
        return Err(Error::InvalidData("Events is not a table".into()));
    };
    let (colnum, desc) = column_descriptions
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == "ACD")
        .ok_or_else(|| Error::InvalidData("no ACD column".into()))?;
    if desc.data_type.typ != fitsio::tables::ColumnDataType::Bit || desc.data_type.repeat != 18 {
        return Err(Error::InvalidData(format!(
            "ACD column is not 18X: {:?}",
            desc.data_type
        )));
    }
    const ROW_BYTES: usize = 3;
    let mut bytes = vec![0u8; num_rows * ROW_BYTES];
    if !bytes.is_empty() {
        let (mut anynul, mut status) = (0, 0);
        unsafe {
            fitsio::sys::ffgcvb(
                fptr.as_raw(),
                (colnum + 1) as i32,
                1,
                1,
                bytes.len() as i64,
                0,
                bytes.as_mut_ptr(),
                &mut anynul,
                &mut status,
            );
        }
        fitsio::errors::check_status(status)?;
    }
    Ok(bytes
        .as_chunks::<ROW_BYTES>()
        .0
        .iter()
        .map(unpack_acd)
        .collect())
}

fn unpack_acd(row: &[u8; 3]) -> [bool; 18] {
    std::array::from_fn(|j| row[j / 8] & (0x80 >> (j % 8)) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fitsio::tables::{ColumnDataType, ColumnDescription};

    #[test]
    fn unpack_is_msb_first() {
        let acd = unpack_acd(&[0b1000_0001, 0b0000_0000, 0b0100_0000]);
        let fired: Vec<usize> = (0..18).filter(|&j| acd[j]).collect();
        assert_eq!(fired, vec![0, 7, 17]);
    }

    /// 写一个各行 ACD 互不相同的小文件，逐行读回。旧实现在第 2 行起就错位。
    #[test]
    fn acd_flags_stay_with_their_rows() {
        let path = std::env::temp_dir().join(format!("blink_acd_test_{}.fits", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let rows: Vec<[bool; 18]> = (0..7)
            .map(|r| std::array::from_fn(|j| (j * 7 + r * 3) % 5 == 0 || j == r))
            .collect();
        {
            let mut f = fitsio::FitsFile::create(&path).open().unwrap();
            let cols: Vec<_> = [
                ("Time", ColumnDataType::Double, 1),
                ("Det_ID", ColumnDataType::Byte, 1),
                ("Channel", ColumnDataType::Byte, 1),
                ("Pulse_Width", ColumnDataType::Byte, 1),
                ("ACD", ColumnDataType::Bit, 18),
                ("Event_Type", ColumnDataType::Byte, 1),
            ]
            .into_iter()
            .map(|(n, t, r)| {
                ColumnDescription::new(n)
                    .with_type(t)
                    .that_repeats(r)
                    .create()
                    .unwrap()
            })
            .collect();
            let hdu = f.create_table("Events", &cols).unwrap();
            let n = rows.len();
            hdu.write_col(
                &mut f,
                "Time",
                &(0..n).map(|i| i as f64).collect::<Vec<_>>(),
            )
            .unwrap();
            for name in ["Det_ID", "Channel", "Pulse_Width", "Event_Type"] {
                hdu.write_col(&mut f, name, &vec![0u8; n]).unwrap();
            }
            for (r, flags) in rows.iter().enumerate() {
                let mut larray: Vec<i8> = flags.iter().map(|&b| b as i8).collect();
                let mut status = 0;
                unsafe {
                    fitsio::sys::ffpclx(
                        f.as_raw(),
                        5,
                        r as i64 + 1,
                        1,
                        18,
                        larray.as_mut_ptr(),
                        &mut status,
                    );
                }
                fitsio::errors::check_status(status).unwrap();
            }
        }
        let file = EventFile::new(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(file.acd, rows);
    }
}
