/*
Filename: svom_att_250101_00_v00.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PrimaryHDU    1 PrimaryHDU      39   ()
  1  Quaternion    1 BinTableHDU     72   2069R x 12C   [1D, 1E, 1E, 1E, 1E, 1E, 1E, 1E, 1B, 1J, 1B, 1B]
*/

//! # SVOM 姿态四元数的约定（2026-09-10 核实）
//!
//! 文件头里没有任何一条注释说明四元数的分量顺序与旋转方向，所以下面这套约定
//! 是从官方标定软件读出来、再用两条互相独立的检验钉死的。
//!
//! **约定**：
//!
//! 1. `Q0` 是**标量分量**，`(Q1, Q2, Q3)` 是矢量部分。
//! 2. 该四元数表示**卫星本体系 → J2000（ICRS）惯性系**的旋转：
//!    `v_J2000 = R(q) · v_body`。
//! 3. 同表的 `wx/wy/wz` 是**卫星本体系**下的角速度（rad/s），不是载荷系。
//! 4. GRM 载荷系与卫星系差一个固定旋转
//!    `M = [[0,0,1], [0,-1,0], [1,0,0]]`（`+Xs = +Zp`、`+Ys = −Yp`、`+Zs = +Xp`），
//!    即 `v_payload = M · v_body`。方向相关响应（CALDB `mc_rsp`）的
//!    `(θ, φ)` 就定义在载荷系里：θ 从 +Z 量起，φ 从 +X 转向 +Y。
//!
//! 于是由天球坐标求入射角的完整链条是
//! `v_payload = M · R(q)⁻¹ · v_J2000`，θ = arccos(v_z)、φ = atan2(v_y, v_x)。
//!
//! **依据一（官方软件）**：`/gecamfs/SVOM/soft/CALDB/software` 里
//! `RSP_Generator/user_inputs.py` 读 att 表时写的是 `quadw = att_data['q0']`、
//! `quadx = att_data['q1']` …，`CommonFunc/coordinate.py` 的 `quat()` 把它们
//! 排成 `[q1, q2, q3, q0]` 交给 `scipy` 的 `Rotation.from_quat`（标量在尾），
//! `radec_to_thetaphi()` 再做 `(R·M).inv().apply(v_J2000)`。
//!
//! **依据二（表内自洽，四天实测）**：att 表自带角速度 `wx/wy/wz`。把四元数按某个
//! 约定装成旋转矩阵，相邻采样点的相对旋转除以 Δt 必须等于表里的角速度。惯性指向
//! 时四元数每秒只变 1e-6 rad、贴着 float32 量化底，没有信号；只取机动样本
//! （|w| > 1e-3 rad/s，转速 0.3–1.3 deg/s）后，上面这套约定的相对残差中位是
//! 1.7–2.3%，逐分量相关 +0.996…+0.999，而另外七种组合（标量放尾、旋转反向、
//! 角速度取惯性系或载荷系）的残差是它的 11–29 倍。2025-09-19 / 2026-02-26 /
//! 2024-12-05 / 2025-01-15 四天各自独立给出同一个答案。
//!
//! **依据三（外部物理，GRB 250919A）**：Fermi-LAT 把 GRB 250919A 定在
//! RA = 298.52°、Dec = −48.83°（误差半径 0.08°）。三个 GRD 都装在离载荷 +Z 轴
//! 30° 处、方位相隔 120°（G01 φ=0°、G02 φ=240°、G03 φ=120°），所以源的方位角
//! 直接决定谁最亮。按上面的链条算出的入射方向是 θ = 83.7°、φ = 245.3°，几乎正对
//! G02；把 CALDB 的方向相关响应折过幂律谱，预测三路份额 0.159 / 0.759 / 0.082，
//! 实测（扣本底、逐探头扣死时间）0.198 / 0.739 / 0.064，χ² = 179；其余七种约定
//! 全都预测成 G01 或 G03 最亮，最好的一个 χ² = 4447（差 25 倍）。
//!
//! **只存矢量部分是无损的**：`blink_core::Attitude` 只有三个分量（这个形状是
//! HXMT 定的——HXMT 1K 的 `ATT_Quater` 表本身就只有 Q1/Q2/Q3）。SVOM 的 `Q0`
//! 实测恒为非负（八天 1 137 637 个样本，负值 0 个，最小 6.9e-5），而 q 与 −q
//! 表示同一个旋转，所以标量分量可以由 `q0 = +sqrt(1 − q1² − q2² − q3²)` 唯一
//! 复原。**用的时候必须自己补这一步**，别把这三个数当成 (Q0,Q1,Q2)。
//! 唯一的隐患是 Q0 → 0（转角趋近 180°）时 `1 − |v|²` 会被 float32 的舍入吃掉，
//! 实测八天里最小的 Q0 是 6.9e-5，只此一个样本进到 1e-4 以内。

use blink_core::types::{Attitude, MissionElapsedTime, TemporalState, Trajectory};

use crate::types::SvomGrm;

pub struct AttFile {
    quaternion: QuaternionHdu,
}

impl AttFile {
    pub fn from_fits_file(path: &str) -> Result<Self, fitsio::errors::Error> {
        let mut fptr = fitsio::FitsFile::open(path)?;

        let quaternion = QuaternionHdu::from_fptr(&mut fptr)?;

        Ok(Self { quaternion })
    }
}

struct QuaternionHdu {
    time: Vec<f64>,
    // Q0 是标量分量，不进 `Attitude`：它恒为非负，可由矢量部分复原（见文件头注释）。
    // q0: Vec<f32>,
    q1: Vec<f32>,
    q2: Vec<f32>,
    q3: Vec<f32>,
    // wx: Vec<f32>,
    // wy: Vec<f32>,
    // wz: Vec<f32>,
    // slew_stat: Vec<u8>,
    // target_id: Vec<i32>,
    // quality: Vec<u8>,
    // att_ref: Vec<u8>,
}

impl QuaternionHdu {
    fn from_fptr(fptr: &mut fitsio::FitsFile) -> Result<Self, fitsio::errors::Error> {
        let quaternion = fptr.hdu("Quaternion")?;

        let time = quaternion.read_col::<f64>(fptr, "TIME")?;
        // let q0 = quaternion.read_col::<f32>(fptr, "Q0")?;
        let q1 = quaternion.read_col::<f32>(fptr, "Q1")?;
        let q2 = quaternion.read_col::<f32>(fptr, "Q2")?;
        let q3 = quaternion.read_col::<f32>(fptr, "Q3")?;
        // let wx = quaternion.read_col::<f32>(fptr, "wx")?;
        // let wy = quaternion.read_col::<f32>(fptr, "wy")?;
        // let wz = quaternion.read_col::<f32>(fptr, "wz")?;
        // let slew_stat = quaternion.read_col::<u8>(fptr, "slew_stat")?;
        // let target_id = quaternion.read_col::<i32>(fptr, "TargetID")?;
        // let quality = quaternion.read_col::<u8>(fptr, "Quality")?;
        // let att_ref = quaternion.read_col::<u8>(fptr, "AttRef")?;

        Ok(Self {
            time,
            q1,
            q2,
            q3,
            // wx,
            // wy,
            // wz,
            // slew_stat,
            // target_id,
            // quality,
            // att_ref,
        })
    }
}

impl From<&AttFile> for Trajectory<MissionElapsedTime<SvomGrm>, Attitude> {
    fn from(att_file: &AttFile) -> Self {
        let points = att_file
            .quaternion
            .time
            .iter()
            .zip(att_file.quaternion.q1.iter())
            .zip(att_file.quaternion.q2.iter())
            .zip(att_file.quaternion.q3.iter())
            .map(|(((t, q1), q2), q3)| TemporalState {
                timestamp: MissionElapsedTime::new(*t),
                // `Attitude` 的三个字段是四元数的**矢量部分**，标量分量
                // q0 = +sqrt(1 − q1² − q2² − q3²)，见文件头注释。
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
