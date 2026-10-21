//! 曝光的活时间：与搜索同一口径。
//!
//! 搜索（`search.rs`）先按机箱求 GTI 交集（`gti_from_events_by_box`），只在交集里
//! 找候选，再把落在饱和窗（±1 s）里的候选整个丢掉。所以能出 TGF 的活时间是
//! **GTI 交集减去饱和窗**；`Coverage` 的 `span − masked` 只扣了饱和、没扣 GTI 之外，
//! 不能当曝光用。

use super::Chunk;
use crate::algorithms::live_time;
use crate::types::HxmtHe;
use blink_core::traits::Event as _;
use blink_core::types::{ExposureSample, MissionElapsedTime, Position, Trajectory};

/// 有序不重叠的区间 `segments` 减去有序不重叠的 `mask`
pub(crate) fn subtract(segments: &[[f64; 2]], mask: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    let mut k = 0usize;
    for &[a, b] in segments {
        let mut start = a;
        while k < mask.len() && mask[k][1] <= start {
            k += 1;
        }
        let mut j = k;
        while j < mask.len() && mask[j][0] < b {
            if mask[j][0] > start {
                out.push([start, mask[j][0]]);
            }
            start = start.max(mask[j][1]);
            j += 1;
        }
        if start < b {
            out.push([start, b]);
        }
    }
    out
}

impl Chunk {
    /// 搜索实际用到的活时间段（MET 秒）：机箱 GTI 交集减去饱和窗
    pub fn live_segments(&self) -> Vec<[f64; 2]> {
        let events = self.event_file.into_iter().filter(|e| e.keep()).collect::<Vec<_>>();
        let gti: Vec<[f64; 2]> = live_time::gti_from_events_by_box(
            &events,
            [self.span[0], self.span[1]],
            live_time::DEFAULT_FALSE_GAP_BUDGET,
        )
        .into_iter()
        .map(|[a, b]| [a.met(), b.met()])
        .collect();
        let saturation: Vec<[f64; 2]> = self
            .saturation_intervals()
            .iter()
            .map(|(a, b)| [a.met(), b.met()])
            .collect();
        subtract(&gti, &saturation)
    }

    pub(super) fn exposure_samples_impl(&self, step: f64) -> Vec<ExposureSample> {
        let positions = Trajectory::<MissionElapsedTime<HxmtHe>, Position>::from(&self.orbit_file);
        let mut samples = Vec::new();
        for [a, b] in self.live_segments() {
            let mut t = a;
            while t < b {
                // 每个采样点代表 [t, t + step) 与段的交集
                let seconds = (b - t).min(step);
                let met = t + seconds / 2.0;
                let position = positions
                    .interpolate(MissionElapsedTime::new(met))
                    .map(|p| p.state);
                samples.push(ExposureSample { met, seconds, position });
                t += step;
            }
        }
        samples
    }
}

#[cfg(test)]
mod tests {
    use super::subtract;

    #[test]
    fn subtracting_masks_keeps_the_rest() {
        let segments = [[0.0, 10.0], [20.0, 30.0]];
        // 掩膜跨段、在段内、贴边、在段外
        let mask = [[-5.0, 1.0], [4.0, 6.0], [9.0, 21.0], [29.0, 29.5], [40.0, 50.0]];
        assert_eq!(
            subtract(&segments, &mask),
            vec![[1.0, 4.0], [6.0, 9.0], [21.0, 29.0], [29.5, 30.0]]
        );
        assert_eq!(subtract(&segments, &[]), segments.to_vec());
        assert!(subtract(&segments, &[[-1.0, 100.0]]).is_empty());
    }
}
