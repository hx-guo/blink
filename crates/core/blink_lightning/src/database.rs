use crate::types::Lightning;
use chrono::prelude::*;
use rusqlite::{Connection, params};
use std::env;

thread_local! {
    // 每个线程持有独立的只读连接：SQLite 允许多读者并发，用线程本地连接（而非
    // 全局 Mutex<Connection>）让 filter 的百万级查询能真正并行，而不是串行等锁。
    static LIGHTNING_CONNECTION: Connection = open_connection();
}

fn open_connection() -> Connection {
    let path = env::var("WWLLN_DB_PATH")
        .unwrap_or_else(|_| String::from("/Volumes/Graphite/WWLLN/WWLLN.db"));
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let conn =
        Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    // Set a longer busy timeout (e.g., 30 seconds = 30000 ms)
    conn.busy_timeout(std::time::Duration::from_secs(30))
        .unwrap();
    // 用 mmap 直接映射数据库读取，绕过 SQLite 全局页缓存（pcache1）那把互斥锁——
    // filter 用几十个线程各开一个连接跑百万级查询时，该锁的争用是主要瓶颈
    // （perf 显示 ~40% 时间在 pthread_mutex_lock/pcache1）。实际驻留由 OS 页
    // 缓存按热度管理，映射本身只花地址空间。
    //
    // 但地址空间不是无限的：映射多少要**按环境算**，不能写死。见 `mmap_budget`。
    // 这条 pragma 只是性能提示，设不上就退回 SQLite 自己的页缓存，**所以不 unwrap**
    // ——为了快而让程序在受限环境里起不来，是本末倒置。
    let _ = conn.pragma_update(None, "mmap_size", mmap_budget(size, address_space_limit()));
    conn
}

/// mmap 映射多少字节。
///
/// 先前这里写死 512 GB（比 422 GB 的库还大一点），在受限节点上见过开库处
/// core dump。**但那个故障的机制不是"写死"**——见下面"这个函数目前是空转的"。
/// 复盘时另有一条实测口径要记住：`hep_sub -mem 20000` **并不压地址空间**，
/// 那一档下 `ulimit -v` 仍然是 95 GB；要造这个故障条件只能用 `ulimit -v` 直接设。
///
/// 两条上界：
/// * **库文件本身的大小**——映射比文件还大的区间没有任何意义。
/// * **地址空间上限的四分之一**——留出余量给堆、线程栈和别的映射。
///
/// # 这个函数目前是空转的，除数也是错的
///
/// 2026-09-11 实测（GECAM-C 单日候选、同一份源码编出只差这一行的两个二进制、
/// 12 次运行）：
///
/// * **SQLite 自己把 `mmap_size` 封在 `SQLITE_MAX_MMAP_SIZE = 0x7FFF0000`（2 GiB）**。
///   实测每条映射恰好 2,097,088 KB，写死 512 GB 与按 `RLIMIT_AS` 算 23.75 GB
///   落到内核那里是同一个数。所以在 `RLIMIT_AS >= 8 GB` 的环境里，本函数改不了
///   任何行为：12 次运行全部退出码 0，`tgfs.json` 逐字节同 md5，`VmPeak` 差 4 KB。
/// * **进程的地址空间占用是 `(线程数 + 1) × 2 GiB`**，与本函数给多大无关。
///   `+1` 是主线程为 `coverage()` 开的那条连接。8 线程实测 9 条映射、18.0 GiB。
/// * **除数应当是线程数，不是 4。** 下面那句"同一个文件的映射在进程里是共享的，
///   所以按一份算"是错的：物理页共享，**虚拟地址空间不共享**，而 `RLIMIT_AS`
///   数的正是后者。实测在同一进程里连续 mmap 同一文件 4 次、每次 5 GB，
///   `VmSize` 从 246,756 KB 长到 19,778,020 KB，一份不少地各占一份。
///   要挡住耗尽需要 `(n+1) × budget < AS`，按 `AS/4` 给预算等于假定 `n <= 2`，
///   而农场节点上 `available_parallelism()` 能到几十。
///
/// **在复量 mmap 到底买到多少性能之前不动这里的公式**：库 422 GB 而 SQLite 最多
/// 映射前 2 GiB（0.47%，而且是文件头部不是热点页），所以当初 perf 量到的
/// pcache1 锁争用被它消掉多少是未知数。若收益很小，正解是把 mmap 整个去掉而不是
/// 换一个除数。
fn mmap_budget(file_bytes: u64, address_space: Option<u64>) -> i64 {
    let budget = match address_space {
        Some(limit) => file_bytes.min(limit / 4),
        None => file_bytes,
    };
    i64::try_from(budget).unwrap_or(i64::MAX)
}

/// 进程的地址空间上限（`RLIMIT_AS`）。取不到或不设限时返回 `None`。
///
/// 走 `/proc/self/limits` 而不是 `getrlimit`，是为了不给这个 crate 引进 `libc`
/// 依赖；非 Linux 上这个文件不存在，当作不设限。
fn address_space_limit() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/limits").ok()?;
    parse_address_space_limit(&text)
}

fn parse_address_space_limit(limits: &str) -> Option<u64> {
    let line = limits
        .lines()
        .find(|line| line.starts_with("Max address space"))?;
    // "Max address space   20971520000   20971520000   bytes"，软限在第一列
    line.split_whitespace()
        .find_map(|field| field.parse::<u64>().ok())
}

/// 库里最早与最晚一条闪电的时刻。
///
/// 用来把"这一刻库里没有闪电"和"这一刻库根本没有数据"分开——库目前止于
/// 2024-12-31，而 SVOM 的候选到 2026 年，落在覆盖外的候选查出来必然是空，
/// 那不是没有闪电。
///
/// 走 `ORDER BY ... LIMIT 1` 而不是 `min()`/`max()`：后者在这个 422 GB 的表上
/// 进不了索引，实测跑 200 s 还没回来；前者是纯索引扫描，几十毫秒。
pub fn coverage() -> (DateTime<Utc>, DateTime<Utc>) {
    LIGHTNING_CONNECTION.with(|connection| {
        let bound = |order: &str| -> DateTime<Utc> {
            let text: String = connection
                .query_row(
                    &format!("SELECT time FROM lightning ORDER BY time {order} LIMIT 1"),
                    [],
                    |row| row.get(0),
                )
                .expect("WWLLN database is empty");
            NaiveDateTime::parse_from_str(&text, "%Y-%m-%d %H:%M:%S%.6f")
                .expect("unparseable time in WWLLN database")
                .and_utc()
        };
        (bound("ASC"), bound("DESC"))
    })
}

pub fn get_lightnings(time_start: DateTime<Utc>, time_end: DateTime<Utc>) -> Vec<Lightning> {
    let time_start_str = time_start.format("%Y-%m-%d %H:%M:%S%.6f").to_string();
    let time_end_str = time_end.format("%Y-%m-%d %H:%M:%S%.6f").to_string();
    LIGHTNING_CONNECTION.with(|connection| {
        let mut statement = connection
            .prepare(
                "
                SELECT
                    time,
                    lat,
                    lon,
                    resid,
                    nstn,
                    energy,
                    energy_uncertainty,
                    estn
                FROM
                    lightning
                WHERE
                    time BETWEEN ?1 AND ?2
                ORDER BY time ASC
                ",
            )
            .unwrap();
        statement
            .query_map(params![time_start_str, time_end_str], |row| {
                Ok(Lightning {
                    time: NaiveDateTime::parse_from_str(
                        &row.get::<_, String>(0).unwrap(),
                        "%Y-%m-%d %H:%M:%S%.6f",
                    )
                    .unwrap()
                    .and_utc(),
                    lat: row.get::<_, f64>(1).unwrap(),
                    lon: row.get::<_, f64>(2).unwrap(),
                    resid: row.get::<_, f64>(3).unwrap(),
                    nstn: row.get::<_, i64>(4).unwrap() as u32,
                    energy: row.get::<_, Option<f64>>(5).unwrap(),
                    energy_uncertainty: row.get::<_, Option<f64>>(6).unwrap(),
                    estn: row.get::<_, Option<i64>>(7).unwrap().map(|x| x as u32),
                })
            })
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_an_address_space_limit_the_whole_file_is_mapped() {
        assert_eq!(mmap_budget(422_000_000_000, None), 422_000_000_000);
    }

    #[test]
    fn a_tight_address_space_limit_caps_the_mapping() {
        // hep_sub -mem 20000 那一档：20 GB 地址空间，映射最多取四分之一
        let limit = 20 * 1024 * 1024 * 1024;
        assert_eq!(mmap_budget(422_000_000_000, Some(limit)), limit as i64 / 4);
    }

    #[test]
    fn a_small_database_is_never_over_mapped() {
        // 映射比文件还大没有意义，哪怕地址空间很宽裕
        assert_eq!(mmap_budget(4096, Some(1 << 40)), 4096);
    }

    #[test]
    fn the_soft_limit_is_the_one_that_binds() {
        let limits = "Limit                     Soft Limit           Hard Limit           Units\n\
                      Max address space         20971520000          unlimited            bytes\n";
        assert_eq!(parse_address_space_limit(limits), Some(20_971_520_000));
    }

    #[test]
    fn an_unlimited_address_space_reads_as_no_limit() {
        let limits = "Max address space         unlimited            unlimited            bytes\n";
        assert_eq!(parse_address_space_limit(limits), None);
    }
}
