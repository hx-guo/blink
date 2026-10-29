//! 一小时的 last_modified 取全部输入文件的最晚修改时间，eng 也在内：eng 晚到时
//! 这一小时先按缺数据排除，eng 到了得让搜索重做这一天。
//!
//! 归档目录由环境变量给出且只读一次，所以单独成一个测试进程。

use blink_core::traits::Chunk as _;
use blink_hxmt_he::types::Chunk;
use chrono::prelude::*;
use std::fs::File;
use std::path::Path;
use std::time::{Duration, SystemTime};

fn touch(path: &Path, modified: SystemTime) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    File::create(path).unwrap().set_modified(modified).unwrap();
}

#[test]
fn late_engineering_file_moves_the_hour() {
    let root = std::env::temp_dir().join(format!("blink_last_modified_{}", std::process::id()));
    let (dir_1b, dir_1k) = (root.join("1B"), root.join("1K"));
    // SAFETY: 单线程，任何读取之前
    unsafe {
        std::env::set_var("HXMT_1B_DIR", &dir_1b);
        std::env::set_var("HXMT_1K_DIR", &dir_1k);
    }

    let epoch = Utc.with_ymd_and_hms(2025, 1, 1, 5, 0, 0).unwrap();
    let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_735_700_000);
    // 1K：事例、轨道、姿态
    let day_1k = dir_1k.join("Y202501/20250101-2758");
    for kind in ["HE-Evt", "Orbit", "Att"] {
        touch(&day_1k.join(format!("HXMT_20250101T05_{kind}_FFFFFF_V1_1K.FITS")), t0);
    }
    // 1B：三箱 sci 与 eng，eng 的 C 箱晚到一天
    let day_1b = dir_1b.join("2025/20250101");
    for (apid, late) in [("0642", false), ("0922", false), ("1686", false), ("0766", false), ("1009", false), ("1781", true)] {
        let modified = if late { t0 + Duration::from_secs(86_400) } else { t0 };
        touch(&day_1b.join(apid).join(format!("HXMT_1B_{apid}_20250101T050000_G000000_000_004.fits")), modified);
    }

    let got = Chunk::last_modified(&epoch).unwrap();
    std::fs::remove_dir_all(&root).ok();
    assert_eq!(got, DateTime::<Utc>::from(t0 + Duration::from_secs(86_400)));
}
