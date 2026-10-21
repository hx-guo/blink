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
        let fptr = fitsio_pure::compat::fitsfile::FitsFile::open(filename)?;

        // HDU 1: Events
        let events = fptr.hdu("Events")?;
        let time = events.read_col::<f64>(&fptr, "Time")?;
        let det_id = events.read_col::<u8>(&fptr, "Det_ID")?;
        let channel = events.read_col::<u8>(&fptr, "Channel")?;
        let pulse_width = events.read_col::<u8>(&fptr, "Pulse_Width")?;

        let acd = read_acd(&fptr)?;

        let event_type = events.read_col::<u8>(&fptr, "Event_Type")?;
        // let flag = events.read_col::<u8>(&fptr, "FLAG")?;

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
/// 不能用 `read_col`：位列不是数值列。早先用 cfitsio 的 `read_col::<u32>`，它对位列按字节
/// 跨行连续填，第 i 个事例拿到的是整列第 i 个字节（约第 i/3 行），标志错配到别的事例上。
/// 这里走 fitsio-pure 的原生接口 `read_binary_column`，按行取出原始字节。
fn read_acd(fptr: &fitsio_pure::compat::fitsfile::FitsFile) -> Result<Vec<[bool; 18]>, Error> {
    use fitsio_pure::bintable::{
        BinaryColumnData, BinaryColumnType, parse_binary_table_columns, read_binary_column_range,
    };
    use fitsio_pure::hdu::HduInfo;
    use fitsio_pure::value::Value;
    let native = |e: fitsio_pure::error::Error| Error::InvalidData(format!("ACD column: {e}"));

    // 与 compat 的 `hdu("Events")` 同一规则：EXTNAME 不分大小写
    let hdu = fptr
        .parsed()?
        .hdus
        .iter()
        .find(|hdu| {
            hdu.cards.iter().any(|card| {
                card.keyword_str() == "EXTNAME"
                    && matches!(&card.value, Some(Value::String(s)) if s.trim().eq_ignore_ascii_case("Events"))
            })
        })
        .ok_or_else(|| Error::InvalidData("no Events HDU".into()))?;
    let HduInfo::BinaryTable { tfields, naxis2, .. } = hdu.info else {
        return Err(Error::InvalidData("Events is not a binary table".into()));
    };
    let columns = parse_binary_table_columns(&hdu.cards, tfields).map_err(native)?;
    let index = columns
        .iter()
        .position(|c| c.name.as_deref().map(str::trim) == Some("ACD"))
        .ok_or_else(|| Error::InvalidData("no ACD column".into()))?;
    let column = &columns[index];
    if column.col_type != BinaryColumnType::Bit || column.repeat != 18 {
        return Err(Error::InvalidData(format!(
            "ACD column is not 18X: {}{:?}",
            column.repeat, column.col_type
        )));
    }
    // X 列读出来是每行一个 Vec<u8>，每行 3 字节数据要多背约 50 字节的堆开销；
    // 一小时上千万行时整列一次读完要多占 1–2 GB，所以分块读、边读边解
    const CHUNK_ROWS: usize = 1 << 20;
    let mut acd = Vec::with_capacity(naxis2);
    for start in (0..naxis2).step_by(CHUNK_ROWS) {
        let rows = CHUNK_ROWS.min(naxis2 - start);
        let BinaryColumnData::Bit(chunk) =
            read_binary_column_range(fptr.data(), hdu, index, start, rows).map_err(native)?
        else {
            return Err(Error::InvalidData("ACD column did not read as bits".into()));
        };
        for row in &chunk {
            let bytes = <&[u8; 3]>::try_from(row.as_slice())
                .map_err(|_| Error::InvalidData(format!("ACD row of {} bytes", row.len())))?;
            acd.push(unpack_acd(bytes));
        }
    }
    Ok(acd)
}

fn unpack_acd(row: &[u8; 3]) -> [bool; 18] {
    std::array::from_fn(|j| row[j / 8] & (0x80 >> (j % 8)) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fitsio_pure::bintable::{
        BinaryColumnData, BinaryColumnDescriptor, BinaryColumnType, build_binary_table_cards,
        serialize_binary_table,
    };
    use fitsio_pure::header::{Card, serialize_header};
    use fitsio_pure::primary::build_primary_header;
    use fitsio_pure::value::Value;

    #[test]
    fn unpack_is_msb_first() {
        let acd = unpack_acd(&[0b1000_0001, 0b0000_0000, 0b0100_0000]);
        let fired: Vec<usize> = (0..18).filter(|&j| acd[j]).collect();
        assert_eq!(fired, vec![0, 7, 17]);
    }

    /// 写一个 1K Evt 结构的小文件（Events 表，ACD 为 18X）
    fn write_events(path: &std::path::Path, rows: &[[bool; 18]]) {
        let n = rows.len();
        let column = |name: &str, col_type: BinaryColumnType, repeat: usize, byte_width: usize| {
            BinaryColumnDescriptor {
                name: Some(name.into()),
                repeat,
                col_type,
                byte_width,
                tdim: None,
            }
        };
        let columns = vec![
            column("Time", BinaryColumnType::Double, 1, 8),
            column("Det_ID", BinaryColumnType::Byte, 1, 1),
            column("Channel", BinaryColumnType::Byte, 1, 1),
            column("Pulse_Width", BinaryColumnType::Byte, 1, 1),
            column("ACD", BinaryColumnType::Bit, 18, 3),
            column("Event_Type", BinaryColumnType::Byte, 1, 1),
        ];
        // FITS 位列：每行 ceil(18/8) = 3 字节，第 j 位在第 j/8 字节、自高位起第 j%8 位
        let bits = rows
            .iter()
            .map(|flags| {
                let mut bytes = vec![0u8; 3];
                for (j, &on) in flags.iter().enumerate() {
                    if on {
                        bytes[j / 8] |= 0x80 >> (j % 8);
                    }
                }
                bytes
            })
            .collect();
        let data = vec![
            BinaryColumnData::Double((0..n).map(|i| i as f64).collect()),
            BinaryColumnData::Byte(vec![0; n]),
            BinaryColumnData::Byte(vec![0; n]),
            BinaryColumnData::Byte(vec![0; n]),
            BinaryColumnData::Bit(bits),
            BinaryColumnData::Byte(vec![0; n]),
        ];
        let mut cards = build_binary_table_cards(&columns, n, 0).unwrap();
        cards.push(Card {
            keyword: *b"EXTNAME ",
            value: Some(Value::String("Events".into())),
            comment: None,
        });
        let mut bytes = serialize_header(&build_primary_header(8, &[]).unwrap()).unwrap();
        bytes.extend(serialize_header(&cards).unwrap());
        bytes.extend(serialize_binary_table(&columns, &data, n).unwrap());
        std::fs::write(path, bytes).unwrap();
    }

    /// 各行 ACD 互不相同的小文件，逐行读回。cfitsio 的 `read_col` 从第 2 行起就错位。
    #[test]
    fn acd_flags_stay_with_their_rows() {
        let path = std::env::temp_dir().join(format!("blink_acd_test_{}.fits", std::process::id()));
        let rows: Vec<[bool; 18]> = (0..7)
            .map(|r| std::array::from_fn(|j| (j * 7 + r * 3) % 5 == 0 || j == r))
            .collect();
        write_events(&path, &rows);
        let file = EventFile::new(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(file.acd, rows);
    }
}
