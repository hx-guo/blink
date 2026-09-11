//! GECAM（引力波暴高能电磁对应体全天监测器）三颗星的 TGF 搜索支持。
//!
//! A、B 是 2020-12-10 一起发射的双星，各 25 路 GRD（LaBr₃）+ 8 路 CPD；
//! C（载荷代号 HEBS，搭在 SATech-01 上）12 路 GRD + 2 路 CPD。三者的事例
//! 产品是同一套格式，差别只在归档路径、文件名前缀和 MET 历元，所以共用这
//! 一个 crate，按 [`types::instrument::Satellite`] 类型参数分星。
pub mod io;
pub mod types;
