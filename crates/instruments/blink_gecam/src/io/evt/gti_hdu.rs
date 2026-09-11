//! GTI（好时间区间）。
//!
//! GECAM 的小时文件不覆盖整点到整点：抽查 GECAM-C 某小时的 GTI 只有 1763 s /
//! 3700 s，GECAM-A 某小时 1303 s / 3600 s，一小时常被切成一到四段。缺口是
//! SAA 穿越与过境安排造成的。曝光核算按 GTI 算，事例流也按同一份 GTI 过滤，
//! 分子分母口径必须一致——SVOM/GRM 上就是因为只在分母上用 GTI，紧挨缺口的
//! 本底窗伸进空区把均值压低，造出过整整两类假信号。

pub(super) struct GtiHdu {
    pub start: Vec<f64>,
    pub stop: Vec<f64>,
}

impl GtiHdu {
    pub fn from_fptr(fptr: &mut fitsio::FitsFile) -> Result<Self, fitsio::errors::Error> {
        let gti = fptr.hdu("GTI")?;
        Ok(Self {
            start: gti.read_col::<f64>(fptr, "START")?,
            stop: gti.read_col::<f64>(fptr, "STOP")?,
        })
    }

    /// `time` 是否落在某个 GTI 段内（闭区间，与 `seconds_within` 同一口径）。
    /// 段数只有个位数，线性扫描即可。
    pub fn contains(&self, time: f64) -> bool {
        self.start
            .iter()
            .zip(self.stop.iter())
            .any(|(start, stop)| time >= *start && time <= *stop)
    }

    /// GTI 与 `[from, to]` 的交集长度（秒）。
    pub fn seconds_within(&self, from: f64, to: f64) -> f64 {
        self.start
            .iter()
            .zip(self.stop.iter())
            .map(|(start, stop)| (stop.min(to) - start.max(from)).max(0.0))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gti(segments: &[(f64, f64)]) -> GtiHdu {
        GtiHdu {
            start: segments.iter().map(|(s, _)| *s).collect(),
            stop: segments.iter().map(|(_, e)| *e).collect(),
        }
    }

    #[test]
    fn a_segment_inside_the_window_counts_in_full() {
        assert_eq!(gti(&[(100.0, 400.0)]).seconds_within(0.0, 3600.0), 300.0);
    }

    #[test]
    fn segments_are_clipped_to_the_window() {
        // 位姿表一小时有 3700 个采样点，事例文件同样越过整点：越界那截不能
        // 计入本小时，否则重叠段会被相邻两小时各记一次，曝光分母偏大。
        assert_eq!(gti(&[(-100.0, 3600.0)]).seconds_within(0.0, 3600.0), 3600.0);
        assert_eq!(gti(&[(3000.0, 4000.0)]).seconds_within(0.0, 3600.0), 600.0);
    }

    #[test]
    fn several_segments_add_up() {
        // 抽查的 GECAM-C 小时就是四段
        let hour = gti(&[
            (0.0, 170.0),
            (1376.0, 2266.0),
            (2631.0, 3090.0),
            (3455.0, 3600.0),
        ]);
        assert_eq!(
            hour.seconds_within(0.0, 3600.0),
            170.0 + 890.0 + 459.0 + 145.0
        );
    }

    #[test]
    fn a_segment_outside_the_window_contributes_nothing() {
        // 负的交集必须夹到 0，不能把「离窗多远」当成曝光减掉
        assert_eq!(gti(&[(5000.0, 6000.0)]).seconds_within(0.0, 3600.0), 0.0);
        assert_eq!(
            gti(&[(5000.0, 6000.0), (0.0, 500.0)]).seconds_within(0.0, 3600.0),
            500.0
        );
    }

    #[test]
    fn an_empty_gti_is_zero_exposure_and_contains_nothing() {
        assert_eq!(gti(&[]).seconds_within(0.0, 3600.0), 0.0);
        assert!(!gti(&[]).contains(100.0));
    }

    #[test]
    fn the_gap_between_segments_is_outside() {
        let hour = gti(&[(0.0, 1000.0), (2500.0, 3600.0)]);
        assert!(hour.contains(500.0));
        assert!(!hour.contains(1000.4));
        assert!(!hour.contains(2499.5));
        assert!(hour.contains(3000.0));
    }

    #[test]
    fn segment_bounds_are_inclusive() {
        let hour = gti(&[(0.0, 1000.0)]);
        assert!(hour.contains(0.0));
        assert!(hour.contains(1000.0));
        assert!(!hour.contains(-1e-9));
        assert!(!hour.contains(1000.0 + 1e-9));
    }
}
