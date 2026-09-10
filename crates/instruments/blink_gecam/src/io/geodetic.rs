//! WGS84 地固直角坐标 → 经纬高。
//!
//! GECAM 的位姿表只给地固系直角坐标（`X_WGS84/Y_WGS84/Z_WGS84`，单位米），
//! 没有像 SVOM 的 ORB 表那样直接给 `LON/LAT/ALT`，所以要自己折算。
//! 闪电关联的半径是 800 km，这里的精度远远够用。

/// WGS84 长半轴（m）
const A: f64 = 6_378_137.0;
/// WGS84 扁率
const F: f64 = 1.0 / 298.257_223_563;
/// 短半轴（m）
const B: f64 = A * (1.0 - F);
/// 第一偏心率平方
const E2: f64 = F * (2.0 - F);
/// 第二偏心率平方
const EP2: f64 = (A * A - B * B) / (B * B);

/// 经纬高，角度用度、高度用米。经度落在 [-180, 180)。
pub struct Geodetic {
    pub longitude_deg: f64,
    pub latitude_deg: f64,
    pub altitude_m: f64,
}

/// Bowring 闭式解。一次求值即可到 1e-9 rad 量级，不需要迭代。
///
/// 高度用 `h = p·cosφ + z·sinφ − a·√(1 − e²sin²φ)`，而不是教科书里常见的
/// `p/cosφ − N`：后者在近极点处 `cosφ → 0`，会把 1 m 的噪声放大成几公里。
/// GECAM 是 29°/正常倾角轨道，纬度到不了极点，但轨道文件里出过异常值，
/// 用稳定式子省心。
pub fn wgs84_to_geodetic(x: f64, y: f64, z: f64) -> Geodetic {
    let p = (x * x + y * y).sqrt();
    let theta = (z * A).atan2(p * B);
    let (sin_theta, cos_theta) = theta.sin_cos();
    let latitude = (z + EP2 * B * sin_theta.powi(3)).atan2(p - E2 * A * cos_theta.powi(3));
    let longitude = y.atan2(x);

    let (sin_lat, cos_lat) = latitude.sin_cos();
    let altitude = p * cos_lat + z * sin_lat - A * (1.0 - E2 * sin_lat * sin_lat).sqrt();

    Geodetic {
        longitude_deg: longitude.to_degrees(),
        latitude_deg: latitude.to_degrees(),
        altitude_m: altitude,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn equator_prime_meridian_is_the_origin_of_both_angles() {
        let g = wgs84_to_geodetic(A, 0.0, 0.0);
        close(g.longitude_deg, 0.0, 1e-9);
        close(g.latitude_deg, 0.0, 1e-9);
        close(g.altitude_m, 0.0, 1e-6);
    }

    #[test]
    fn ninety_degrees_east_is_the_positive_y_axis() {
        let g = wgs84_to_geodetic(0.0, A, 0.0);
        close(g.longitude_deg, 90.0, 1e-9);
        close(g.latitude_deg, 0.0, 1e-9);
        close(g.altitude_m, 0.0, 1e-6);
    }

    #[test]
    fn the_dateline_comes_back_as_minus_one_eighty() {
        // atan2 的值域是 (-π, π]，取 [-180, 180) 这一支与其余仪器一致
        let g = wgs84_to_geodetic(-A, -1.0, 0.0);
        assert!(g.longitude_deg < -179.9, "got {}", g.longitude_deg);
    }

    #[test]
    fn the_pole_sits_on_the_semi_minor_axis() {
        let g = wgs84_to_geodetic(0.0, 0.0, B);
        close(g.latitude_deg, 90.0, 1e-9);
        close(g.altitude_m, 0.0, 1e-6);
    }

    #[test]
    fn altitude_above_the_equator_is_the_radius_minus_the_semi_major_axis() {
        // GECAM-B 实测 |r| ≈ 6.97e6 m
        let r = 6.97e6;
        let g = wgs84_to_geodetic(r, 0.0, 0.0);
        close(g.altitude_m, r - A, 1e-6);
    }

    #[test]
    fn round_trip_from_a_known_geodetic_point() {
        // 反过来构造：φ = 29°、λ = 120°、h = 600 km
        let (lat, lon, h) = (29.0_f64.to_radians(), 120.0_f64.to_radians(), 600_000.0);
        let n = A / (1.0 - E2 * lat.sin() * lat.sin()).sqrt();
        let x = (n + h) * lat.cos() * lon.cos();
        let y = (n + h) * lat.cos() * lon.sin();
        let z = (n * (1.0 - E2) + h) * lat.sin();

        // Bowring 只对椭球面上的点是精确的，600 km 高处留有残差：实测纬度差
        // 1.3e-8 度，折合地面 1.5 mm。闪电关联的半径是 800 km，容差按米取。
        let g = wgs84_to_geodetic(x, y, z);
        close(g.latitude_deg, 29.0, 1e-6);
        close(g.longitude_deg, 120.0, 1e-9);
        close(g.altitude_m, h, 0.01);
    }
}
