# `mmap_budget` 的验证：那一行是空转的

结论先放：**`mmap_budget` 在任何 `RLIMIT_AS >= 8 GB` 的环境里改不了任何行为**，
因为 SQLite 自己把 `mmap_size` 封在 2 GiB；进程的地址空间占用是
`(线程数 + 1) × 2 GiB`，与这个函数给多大无关。**它既没有造成损害，也没有修好
它声称要修的那个故障——两个版本在同一个地址空间上限下同时崩。**

日期 2026-09-11，机器 IHEP 农场，数据 GECAM-C 2023-06-15 单日 3385 个候选，
命令 `blink wwlln --instrument gecam-c`，库 `/gecamfs/.../WWLLN.db`（422 GB）。

## 一、对照怎么立的

从**同一份 main 源码**（commit `87c191c`）编两个二进制，**只差 `database.rs` 里
mmap 预算那一行**，其余逐字节相同：

| 二进制 | 那一行 | md5 |
|---|---|---|
| `blink_main` | `mmap_budget(size, address_space_limit())` | `212c372dbe5a84ade593b7d24c2abd3a` |
| `blink_mmap512` | 写死 `512 GiB` | `ac975cc383ecada69dc16f3c6eceff95` |

**所以任何行为差异只能归到这一行。** 不拿改动前的历史 commit 当对照——那与 main
差好几个 commit，崩不崩可能是别的原因。

**故障条件用 `ulimit -v` 直接造，并在日志里打出来确认生效。** 这一条是上一轮
验证做不出来的根本原因：**`hep_sub -mem 20000` 并不压地址空间**，那一档下
`ulimit -v` 仍然是 100000000 KB（95 GB），所以"复现故障"的那两次根本没有复现。

## 二、第一轮：结果一致这一半成立

2 个版本 × `ulimit -v` {95 GB, 20 GB} × 线程 {1, 4, 8}（`taskset` 控制）= 12 次：

| 线程 | EXIT | VmPeak | WWLLN.db 映射条数 | 映射总量 | `tgfs.json` md5 |
|---|---|---|---|---|---|
| 1 | 0 | 4,274,660 KB | 2 | 4,194,176 KB | `4d7a9aaa…` |
| 4 | 0 | 10,768,724 KB | 5 | 10,485,440 KB | `4d7a9aaa…` |
| 8 | 0 | 19,427,476 KB | 9 | 18,873,792 KB | `4d7a9aaa…` |

**12 次全部 EXIT=0，12 个 `tgfs.json` 逐字节同 md5，两个版本的 VmPeak 差 4 KB。**
逐字节同 md5 比逐行 diff 强，这一半可以结案。

## 三、根因：SQLite 自己封顶 2 GiB

**每条映射恰好 2,097,088 KB = `0x7FFF0000`**，这是 SQLite 的编译期上限
`SQLITE_MAX_MMAP_SIZE`。**`mmap_size` 这个 pragma 给多大都会被封在这里**，所以
写死 512 GB 与按 `RLIMIT_AS` 算 23.75 GB 落到内核那里是同一个数。

**映射条数 = 线程数 + 1**（`+1` 是主线程为 `coverage()` 开的那条连接）。
`blink_wwlln` 每线程一条 thread-local 连接，各自映射一次。

**所以地址空间占用 = `(线程数 + 1) × 2 GiB`，与这个函数无关。**

## 四、那条注释是错的：映射不共享地址空间

注释原文：`同一个文件的映射在进程里是共享的，所以这里按一份算`——**这正是
`/4` 那个除数的来源**。实测（`ulimit -v 20000000`，同一进程里连续 mmap 同一个
文件 4 次、每次 5 GB）：

```
起始 VmSize        246,756 KB
第 1 次映射后    5,129,572 KB
第 2 次映射后   10,012,388 KB
第 3 次映射后   14,895,204 KB
第 4 次映射后   19,778,020 KB
```

**一份不少地各占一份。** 物理页共享，虚拟地址空间不共享，而 `RLIMIT_AS` 数的
正是后者。要挡住耗尽需要 `(n+1) × budget < AS`，按 `AS/4` 给预算等于假定
**线程数不超过 2**。

## 五、第二轮：两个版本在同一条线上同时崩

线程数控制换了两次。**`taskset` 是有效的，但 `nproc` 会骗人**——农场作业里
`OMP_NUM_THREADS=1` 而 `nproc` 认这个变量，所以 `taskset -c 0 nproc` 仍打印 8；
真正的可见核数要看 `sched_getaffinity`。实测一个不带 `-cpu` 的农场作业：

```
nproc               = 1
OMP_NUM_THREADS     = 1
Cpus_allowed_list   = 0-23
sched_getaffinity 个数 = 24
```

**`available_parallelism()` 跟的是 affinity / cgroup 配额，不认 `OMP_NUM_THREADS`。**
所以一个看起来"只有 1 个核"的农场作业实际会起约 24 个线程，
**要 25 × 2 GiB = 50 GiB 地址空间**——这才是生产环境里真正的失效模式。

固定这样一个作业（未加 `-cpu`），扫 `ulimit -v` 从 20 GB 降到 4 GB，
两个版本各一遍，**18 次全部崩，没有一次不同**：

| `ulimit -v` | `blink_mmap512` | `blink_main` |
|---|---|---|
| 20 GB | EXIT=134 | EXIT=134 |
| 19.5 GB | EXIT=101 | EXIT=134 |
| 19 GB | EXIT=134 | EXIT=134 |
| 18 GB | EXIT=134 | EXIT=134 |
| 16 GB | EXIT=134 | EXIT=134 |
| 12 GB | EXIT=134 | EXIT=134 |
| 8 GB | EXIT=101 | EXIT=134 |
| 6 GB | EXIT=134 | EXIT=101 |
| 4 GB | EXIT=134 | EXIT=134 |

134 与 101 的差别是崩溃时哪个线程先报，不是结果差别；两者都是非零退出、
都没有 `tgfs.json`。失效签名两个版本也一样：

```
thread 'main' panicked at .../thread/scoped.rs:206:
  failed to spawn thread: Os { code: 11, kind: WouldBlock, ... }
thread '<unnamed>' panicked at crates/core/blink_lightning/src/database.rs:17:
  SqliteFailure(Error { code: OutOfMemory, extended_code: 7 }, ...)
```

**这与最初报告的那个故障签名逐字相同，而新版本照样产生它。**

## 六、按公式算，这一行只在一个空集般的角落里起作用

预算 `min(文件大小, AS/4)` 再被 SQLite 封在 2 GiB。要它改变行为需要
`AS/4 < 2 GiB`，即 `AS < 8 GB`；而要不崩又需要 `(n+1) × AS/4 < AS`，即
`n <= 2`。**两个条件同时成立的区间是"地址空间小于 8 GB 且线程数不超过 2"**，
而农场作业的线程数由 `available_parallelism()` 定，实测是 24。

## 七、公式怎么改还没定，因为前面还有一个问题

**库 422 GB，SQLite 最多映射前 2 GiB（0.47%），而且是文件的头部、不是热点页。**
当初引入 mmap 的理由是 perf 量到约 40% 时间在 `pthread_mutex_lock`/pcache1，
**但那 40% 里被这 2 GiB 消掉多少从来没有复量过**。

- 若收益小：**正解是把 mmap 整个去掉**（`mmap_size` 设 0 或删掉整段预算计算），
  地址空间占用从 `(n+1) × 2 GiB` 变成 0，本文记录的一整类问题一并消失。
- 若收益大：按 `min(文件大小, SQLITE_MAX_MMAP_SIZE, AS × 安全系数 / 线程数)`
  改，线程数取 `available_parallelism()`。

**在复量之前不动公式。** 本轮只改了注释（commit `cd5e25c`），行为一个字节没动。

## 八、两条可迁移的口径

1. **`hep_sub -mem N` 不压地址空间。** 要造 `RLIMIT_AS` 受限的故障条件只能用
   `ulimit -v` 直接设，并把 `ulimit -v` 的值打进日志确认生效。
2. **农场作业里 `nproc` 不是可见核数**（它认 `OMP_NUM_THREADS`，condor 设成 1），
   而 Rust 的 `available_parallelism()` 认的是 affinity 与 cgroup 配额。
   **两者能差 24 倍，而线程数决定连接数、连接数决定地址空间。**
   这个集群上控制线程数要靠 `taskset` 或程序自己的参数，别信 `nproc`。

## 九、复现

脚本与全部日志在集群 `/scratchfs2/gecam/guohx/mmapcheck/`：
`run_matrix.sh` / `matrix.log`（第一轮）、`run_threads.sh` / `threads.log`
（线程扫描，因 `nproc` 误导而无效，保留作记录）、`run_aslimit.sh` / `aslimit.log`
（第二轮）、`job_probe_cpu.sh` / `probe_cpu_nocpu.log`（可见核数）。
二进制与 provenance 在 `/scratchfs2/gecam/guohx/gecam_bin/`。
