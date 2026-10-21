# 1B 时间重建算法设计

## 目标

从 Level 1B 原始遥测数据中，为每个物理事件重建绝对 MET（Mission Elapsed Time）。

## 已知条件

### 硬件

1. **数据通路**：探测器 → ASIC → FPGA → FIFO A (M67204H) → MCU (8051) → FIFO B → 1553B → 下传
2. **FIFO A 严格保序**：先写入的先读出，MCU 读出的事件时间严格升序
3. **FIFO A 满时不覆盖**：FPGA 端写入被阻塞，新事件直接丢弃
4. **FIFO A 复位**：MCU 可清空整个 FIFO（`FIFOAFullReset()`），复位后所有缓冲数据丢失
5. **MCU 单线程**：主循环轮询，每次 `HandlePhysicalLVDS()` 从 FIFO A 读 109 个事件打包

### ptime

6. **19-bit 硬件计数器**，分辨率 2μs/tick
7. **模值**：PTIME_MOD = 524288 (2^19)
8. **回绕周期**：WRAP_PERIOD = 524288 × 2μs ≈ 1.048576s
9. **由 FPGA 硬件产生**，MCU 不修改，不可配置
10. **在 FIFO 中保序**：FIFO 内的 ptime 严格单调递增（跨回绕时模递增）

### SEC（秒事件）

11. **硬件每秒注入**一个 SEC 事件到 FIFO，记录 (stime, ptime)
12. **stime**：32-bit 绝对秒计数，`MET = stime + offset`
13. **SEC 提供锚点**：(MET, ptime) 构成时间基准
14. **SEC 也经过 FIFO**：与物理事件一起排队，受同样的拥塞/复位影响

### CCSDS 包

15. **每包 109 个事例**：6 字节包头 + 872 字节载荷 (109×8) + 4 字节 UTC 尾
16. **包内事件来自 FIFO 顺序读出**：ptime 保持升序（模递增），但不保证连续——静默丢数可在包内发生
17. **UTC 尾**：MCU 完成打包时的 GPS 时间（反映 MCU 当前时间，非事件时间）
18. **4-bit CRC**：每个事例 8 字节中最后 4 bit 为校验，1/16 碰撞概率

### 丢数与数据损坏

19. **静默丢数（Silent drops）**：FIFO 满时 FPGA 端写入被阻塞，新事件直接丢弃。MCU 继续读取，读出部分数据后 FIFO 不再满，FPGA 恢复写入。**丢失的事件没有任何标记，无法检测。可以发生在任何时刻，包括单个 CCSDS 包内的 109 个事件之间。**
20. **FIFO 复位丢数**：整个 FIFO 被清空，丢失大量连续事件
21. **CRC 碰撞幽灵**：损坏数据碰巧通过 CRC，产生随机 ptime 的假事件

### 单粒子效应（SEE）

22. 极端 GRB（如 GRB 221009A）期间，高通量粒子可引发单粒子效应，影响 MCU、FPGA 或 FIFO 芯片
23. **数据中观测到的影响**：GRB 221009A 数据中三个 Box 在几乎相同的时间出现 SEC 间隙（10~20s 无有效 SEC），各 Box 的 MCU 独立受影响。UTC tail 分析表明 MCU 在间隙期间仍在运行（持续产出包），SEC 消失是因为 FIFO 拥塞或 SEE 导致 SEC 数据损坏，而非 MCU 崩溃。中间的两段中断（如 T+264→T+274）为 SAA 主动关机。

各 Box SEC 间隙结构不同（MCU 独立受影响）：

| Box | T+249~268 区间结构 |
|-----|-------------------|
| A | 单个 19s 间隙 |
| B | 2s + 3s + 3s + 2s + 8s（多段小间隙）|
| C | 3s + 16s |

24. **数据损坏**：饱和期间 byte[0]=0x5A 的出现率从正常的 0.39% 升至 1.13%。0x5A 是 FIFO A 中事件的起始标记，正常情况下被 MCU 剥离不会出现在 CCSDS 载荷中。升高表明 SEE 可能破坏了 MCU 读取 FIFO 的指针或地址逻辑，导致字节错位，将起始标记混入事件数据
25. MCU 看门狗复位后会执行 `ResetFIFOA = 0x00`（清空 FIFO），因此每次崩溃恢复都伴随 FIFO 复位丢数

### 经验参数

26. **MET_CORRECTION = 4.0s**：1B→1K 经验时间校正，通过 GRB 交叉验证确定

## 核心公式

```
MET = anchor_met + (ptime - anchor_ptime + n × PTIME_MOD) × 2μs + MET_CORRECTION
```

- `anchor`：来自 SEC 的 (met, ptime)
- `ptime`：事件自身的 19-bit 值
- **n（wrap 次数）是唯一的未知量**

## n 的唯一性条件

ptime 只提供模 PTIME_MOD 的余数，不提供商。

**n 可唯一确定，当且仅当从 anchor(SEC) 到该事件之间的所有 ptime 回绕都被观测到。**

这要求 SEC 与事件之间的事件序列是连续的（无丢失）。一旦中间丢了事件，就可能漏掉一次回绕，n 差 ±1，对应 MET 偏差 ±1.048576s。

### 无法保证连续性

事件序列的连续性在任何尺度上都无法从数据本身证明：

- **包内**：静默丢数可以发生在 MCU 读取 FIFO 的过程中（HandlePhysicalLVDS 耗时 ~7ms，期间 FIFO 可能短暂满后恢复），导致同一个 CCSDS 包的 109 个事件之间存在看不见的间隙
- **包间**：FIFO 复位、MCU 忙于其他任务、数据下传丢包等均可导致相邻包之间的事件不连续
- **CRC 错误**：通不过 CRC 的事件被丢弃，等效于数据缺失

**因此，不能仅凭"事件在同一个包内"就认定连续性。**

## 算法

### Step 1: 解析 + CRC 过滤

逐包解析 109 个 8-byte slot，通过 4-bit CRC 区分三类：
- **EVT** — 物理事件，提取 ptime (19-bit) 和 channel
- **SEC** — 秒事件，提取 (stime, ptime)
- **Error** — CRC 不通过，丢弃

CRC 假阴性（真实事件未通过 CRC）存在但极少。GRB 260226A Box A 在 549,661 个 1K 事件中仅有 3 个 1B 缺失（0.0005%），均为 CRC 假拒绝或 dead zone 事件。尝试恢复 CRC 失败事件（将其放入 LIS 候选池）风险过大：饱和期间 CRC 错误率可达 ~50%（8,974,246 CRC 错误 vs 9,523,907 通过），绝大部分是真正的数据损坏，LIS 无法有效区分（随机 ptime 几乎必定能插入密集序列的某个间隙）。

### Step 2: SEC 验证，过滤幽灵 SEC

两阶段过滤：

**Phase 1 — 相位聚类**：计算每个 SEC 的相位 `(ptime - stime × 500000) mod 524288`。真实 SEC 的相位聚集在一个窄带（±200 ticks），幽灵 SEC 的相位随机分布。用滑动窗口找最大相位簇。

**Phase 2 — LIS 升序验证**：簇内 SEC 按打包顺序排列后，stime 必须严格递增（FIFO 保序）。用最长递增子序列 (LIS) 找主序列，不在 LIS 中的为幽灵，剔除。

### Step 3: 对每对相邻 SEC，解算中间事件的 MET

对每对相邻有效 SEC (SEC1, SEC2)，收集它们之间（按文件打包顺序）的所有物理事件。

#### Δstime=1 的 SEC 对：直接 LIS

当 Δstime=1 时，每个事件的 elapsed_fwd 只有唯一解（k=0）：

```
elapsed_fwd = (ptime - SEC1.ptime) mod PTIME_MOD
```

合法条件：`elapsed_fwd ∈ [0, 500000]`（即 1 秒内）。不满足的为 dead zone 事件（~4.6%，对应 ptime 回绕周期与整秒的差值 48.6ms）。

**直接对所有合法事件的 elapsed_fwd 求最长严格递增子序列（LIS）**：
- LIS 中的事件 → 赋 MET
- 不在 LIS 中的事件 → 标记为幽灵（NaN）

MET 从两个 SEC 分别计算，取平均：
```
met_fwd = SEC1.met + elapsed_fwd × 2μs + MET_CORRECTION
met_bwd = SEC2.met - (total_ticks - elapsed_fwd) × 2μs + MET_CORRECTION
MET = (met_fwd + met_bwd) / 2
```

##### 为什么用 LIS 而不是贪心

旧算法使用贪心（逐个扫描，选 > prev 的最小候选）+ LIS 验证。贪心对 CRC 碰撞幽灵非常脆弱：

```
真实事件序列:   ... ef=168027, [静默丢数], ef=182860, ef=183000, ...
幽灵事件:       ef=284834 (CRC 碰撞，随机 ptime)

贪心处理:
  prev=168027 → 幽灵 ef=284834 > prev ✓ → prev=284834
  → ef=182860 < prev=284834 → 死亡
  → ef=183000 < prev=284834 → 死亡
  → ... 所有 ef < 284834 的真实事件级联死亡
```

一个幽灵事件可以制造 ~0.2 秒的数据空洞。

直接 LIS 不受此影响：幽灵的随机 ef 不在最长递增子序列中，被自然排除，不影响其他事件。

##### 实测验证（GRB 260226A Box A）

| | 旧（贪心+LIS） | 新（直接 LIS） |
|---|---|---|
| 解算事件 | 9,493,210 | 9,517,246 |
| T+22.35 缺口 | 0 evt/s | 10,460 evt/s |
| 与 1K 残差 | 2,673 events | 3 events |

#### Δstime>1 的 SEC 对：分组 LIS + UTC 约束

当 Δstime=k (k>1) 时，每个事件有 k 个候选：
```
actual_elapsed = elapsed_fwd + i × PTIME_MOD,  i = 0, 1, ..., k-1
```

**问题本质**：从每个事件的 k 个候选中最多选一个，使选出的 elapsed 严格递增，最大化选中事件数。这是经典的"分组 LIS"问题。

**算法**：修改版 patience sorting，逐事件处理，同一事件的候选按**降序**处理：

```
tails = []
for each event (in file order):
    candidates = [ef + i*PTIME_MOD for valid i], sorted DECREASING
    for c in candidates:
        pos = bisect_left(tails, c)
        if pos == len(tails): tails.append(c)
        else: tails[pos] = c
```

**为什么降序处理**：如果升序处理，较大候选会利用同组较小候选创造的位置——但它们来自同一事件，不能同时选。降序处理时，较小候选只能覆盖较大候选的位置或更早位置，不会利用同组候选的结果。

**为什么回溯不会选中同一事件的多个候选**：处理事件 i 的候选时，较大候选先入 tails 位置 p₁，较小候选后入位置 p₂ ≤ p₁。较小候选的 parent 指向 tail_entry[p₂-1]，此位置尚未被事件 i 的任何候选修改（因为事件 i 的所有候选都在 p₂ 或更高位置）。因此 parent 链始终指向前序事件，不会形成同事件环。

**复杂度**：O(Nk log N)，k 为 Δstime。

**为什么能正确消歧 wrap**：真实事件在时间上密集排列（μs 级间距），错误的 wrap 分配将事件位移 ~1.05s，破坏与邻近事件的单调性。LIS 自然选出正确 wrap 的事件（它们形成最长递增链），错误 wrap 被排除。

##### UTC tail 约束

每个 CCSDS 包的 4 字节 UTC 尾记录 MCU 打包时的 GPS 时间。由于事件先进 FIFO 再被 MCU 读出打包，事件产生时间 ≤ 包的 UTC tail。转化为 elapsed 上界：

```
max_elapsed = (utc_tail + 1 - sec1.met) / 2μs
```

（+1 因为 UTC tail 为整秒截断）。此约束在分组 LIS 前剪枝候选 wrap：

| 包的 UTC（相对 SEC1） | 最大允许 wrap (ds=19) |
|----------------------|---------------------|
| +1s | 1 |
| +5s | 5 |
| +10s | 10 |
| +15s | 14 |
| ≥18s | 18（无约束）|

早期包候选大幅缩减，后期包约束较松。实测中 LIS 已在 UTC 约束范围内选择，输出不变。约束作为安全护栏防止极端情况下的 wrap 误选。

##### wrap 解的唯一性

在当前数据的事件密度下（~10,000 evt/s），分组 LIS 的解是**唯一的**。验证方法：对每个 Δstime>1 的 SEC 对，重跑 LIS 并禁止 wrap 0（强制 elapsed ≥ PTIME_MOD），LIS 长度严格减少——说明不存在等长的替代方案。

原因：事件在文件中按 FIFO 读出顺序排列（文件序）。将边界事件从 wrap 0 移到 wrap 1 会使其 elapsed 增大 ~524288 ticks，与后续文件序中的 wrap 1 事件产生升序冲突，LIS 不得不丢弃部分事件。

| GRB | SEC 对 | ds | LIS (原始) | LIS (skip wrap 0) | 丢失 |
|-----|--------|-----|-----------|-------------------|------|
| 221009A Box A | T+249~268 | 19 | 212,494 | 209,493 | 3,001 |

##### 跨 Box 互相关验证

三个 Box 观测同一天体源，光变曲线应一致。Box B 在 GRB 221009A T+249~268 有更小的 SEC 间隙（Δstime=2~8），时间定位更可靠。以 Box B 为参考：

| 对比 | |A - B| 残差之和 |
|------|-----------------|
| Box A 原始（wrap 0） | 10,862 |
| Box A 偏移 +1 wrap (+1.049s) | 30,572 |

wrap 0 与 Box B 的吻合度最高（残差最小），确认当前分配正确。与 1K 在边界处的 ~1s 偏差是 1K 的问题，不是 1B 的问题。

##### 实测验证

旧算法（贪心+LIS）在 GRB 221009A T+220~270 产生三个 Box 完全不相关的光变曲线。分组 LIS 后三个 Box 的光变曲线形状相关，整体包络跟踪 1K。GRB 260226A Box A 残差保持 3 events（与 Δstime=1 直接 LIS 一致）。

## FIFO 拥塞导致 SEC 与物理事件在文件中分离

### 现象

GRB 260226A Box C：连续 11 个包（pkt 11222-11232）全部为 109 个 SEC 事件、0 个物理事件。加上两端过渡包，共 1329 个 SEC 覆盖 1328 秒（~22 分钟）。

GRB 221009A Box C：pkt 38256 包含 20 个 SEC（覆盖 19 秒）+ 89 个物理事件。

### 原因

饱和期间 FIFO A 持续满载：
- 物理事件产生速率远超 MCU 读出速率
- FIFO 满时 FPGA 阻塞写入，物理事件被**静默丢弃**
- SEC 每秒仅 1 个（远低于物理事件速率），不会被丢弃
- FIFO 中逐渐积累大量 SEC，物理事件越来越少

### 对算法的影响

Step 3 按打包顺序收集"两个 SEC 之间的事件"。当 SEC 被集中读出时：
- 相邻 SEC（如 stime=360, stime=361）在同一个包的相邻 slot，之间 0 个物理事件
- 这段时间的物理事件已被硬件丢弃，不存在于数据中
- 拥塞区两端的过渡事件可正确解析（ptime 证实它们确实属于边界的 1 秒窗口）

**对于 Δstime=1 的情况，拥塞区不会导致误分配——物理事件已在硬件层面丢失，算法正确地不产生输出。**

## 残差分析（与 1K 对比）

### GRB 260226A Box A：3 events 残差

549,658 (1B) vs 549,661 (1K)，差 3 个事件分布在 3 个 0.2s bin 中，每 bin 差 1。原因：

| 位置 | 原因 |
|-----|------|
| T+25~26 | CRC 假拒绝：23 个 CRC 失败 slot 中 1 个是真实事件 |
| T+28~29 | CRC 假拒绝：同上 |
| T+29~30 | Dead zone：elapsed_fwd=524287（差 1 tick 超出 1s 窗口）|

### GRB 221009A Box A：边界偏移

T+249~268 的 19s SEC 间隙中，1B 与 1K 总事件数一致（212,494 vs 212,505），但 1B 比 1K 早 ~1s。这不是 wrap 错误（唯一性已验证、跨 Box 已验证），而是两种时间重建方法的固有差异。

UTC tail 分析证实 MCU 在间隙期间持续运行（包速率从 47/s 降到 3-10/s，无 UTC 跳变），SEC 消失是 FIFO 拥塞/SEE 数据损坏所致。

## 已考虑但放弃的方案

### CRC 失败事件恢复

将 CRC 失败事件放入 LIS 候选池，让单调性约束裁决真假。理论上可行（假拒绝的 ptime 正确，能通过 LIS；真拒绝的 ptime 随机，被 LIS 排除）。

**放弃原因**：饱和期间 CRC 错误率高达 ~50%（~900 万个），LIS 序列密集（gap ~50 ticks），随机 ptime 几乎必定能插入某个 gap。大量损坏事件会泄漏到输出中，严重污染数据。收益仅 3 events / 549,661 = 0.0005%。

### wrap 歧义检测

对 Δstime>1 的 LIS 结果，检测边界事件的 wrap 是否唯一。两种方案：
- **逐事件检测**：检查每个 LIS 成员是否有替代 wrap 可在前后邻居之间。结果：0 歧义（邻居间距 ~μs，任何替代 wrap 偏移 ~1.05s 远超间距）。
- **整块检测**：重跑 LIS 禁止 wrap 0，比较长度。结果：LIS 严格缩短（文件序约束使边界整块偏移不可行）。

**放弃原因**：在当前事件密度下（~10,000 evt/s），LIS 解唯一，检测永远不会触发。

### 静默丢数（Silent drop）检测

FIFO 满时 FPGA 阻塞写入，新到达的事件被静默丢弃，数据中没有任何标记。尝试用包内泊松异常检测识别丢数间隙：对每个包内事件间隔，用短间隔（<1ms）估算本底率 λ，检验每个间隔的泊松概率 log₁₀(p) < -10 则标记为 silent drop。

**放弃原因**：对三个 GRB × 三个 Box 的检测结果进行人工检查，确认绝大部分为误报。根本问题是 **λ 在单包时间跨度内不稳定**——GRB 物理源率在毫秒尺度上变化显著，用包内最密集部分的事件率去评判率自然下降的区域，会将正常的稀疏区误判为丢数。单个包仅 109 个事件（~7ms），统计量不足以区分"源率变化"和"事件丢失"。

此外，即使 silent drop 真实发生，其影响也可忽略：silent drop 发生在 FIFO 接近满的瞬间，随后立即触发 FIFO reset（整包丢失），单次 silent drop 丢失的事件数（~几十个）远小于 FIFO reset gap 丢失的事件数（~数千个）。

**结论**：silent drop 作为硬件机制真实存在，但无法从数据中可靠检测，且对总事件损失的贡献可忽略。文章中应描述此机制并给出上述定量论证。

### 深度饱和包级修正

对未触发 FIFO 复位但 MCU 跟不上的"深度饱和"包，尝试用包内 burst 段（间隔 <1ms 的密集部分）的计数率填充 idle 段（>1ms 的大间隔）。

**放弃原因**：与静默丢数检测同理——触发条件非常窄（邻居率 < MCU 上限但包内 burst 率 ≥ MCU 上限且 gap 占比 > 30%），实际产出极少。饱和重建不应区分"深度饱和"和"普通 FIFO 复位"，统一用 FIFO 复位 gap 的跨 box 参考填充即可。

### post-reset 包率估算 N_lost

旧方案用 post-reset 包的事件率 × gap 时长估算 N_lost，再用参考 box 的形状函数做归一化分配。

**放弃原因**：逻辑自相矛盾。构建 shape function 的前提是 gap 期间事件率时变（GRB 光变在 ms 尺度上可有剧烈变化），但 N_lost 却用 gap 结束时刻的瞬时率乘以时长——等价于假设率恒定。post-reset 包只反映 gap 结束时的率，若 gap 跨 GRB 脉冲的上升/下降沿，会系统性高估或低估。

**当前方案**：校准后的 shape function 各 bin 总和直接作为 N_lost。形状和总数来自同一参考源，逻辑自洽。仅在参考 box 不可用时（所有 box 同时饱和），退化为 post-reset 包率 + 均匀分配。

## FIFO 复位 gap 重建算法

### 概述

对 detect 步骤发现的每个 FIFO reset gap，用其他 box 的同时段事件分布重建丢失的光变曲线。

### 步骤

#### 1. 构建形状函数

将 gap 时间段切成 1ms 的 bin。对每个 bin：

- 查其他 box（参考 box）在同一时段的事件数
- 若参考 box 在该时段也不可信（FIFO reset / 拥塞宽包 / 包内异常间隔），跳过
- 校准系数 k = target 在 gap 前后各 0.5s 的事件**率** / ref 在同区间的事件**率**，双方各自排除自身的 unreliable 区间后统计，用有效时长归一化为率
- 校准后计数 = ref_count × k
- 多个参考 box 取平均

#### 2. 确定 N_lost

- **有参考（≥30% bin 有值）**：空 bin 从最近的有值 bin 线性插值（边缘常数外推），N_lost = round(Σ shape)
- **无参考（所有 box 同时饱和）**：用 pre-reset 和 post-reset 包的事件率做线性插值构建 shape（仅一侧可用时常数外推），N_lost = round(Σ rate_i × bin_width)

#### 3. 分配事件

shape 归一化到 N_lost，每个 bin 按比例分配事件数，bin 内等间距放置。

### 不可信区间检测

参考 box 的事件分布在以下情况下不可信，需排除：

1. **FIFO reset gap**：整包丢失的时间段
2. **拥塞宽包**：包跨时 > 邻居中位跨时 × 3
3. **包内异常间隔**：包内存在泊松概率 log₁₀(p) < -10 的大间隔（λ 从邻居事件率估算）

### 可视化

- `scripts/saturation/plot_reconstruct.py` — 光变曲线级别：观测事件 + 填充事件 vs 1K，残差面板
- `scripts/saturation/plot_fifo_resets.py` — 事件级别：每个 FIFO reset 周围 3 个 box 的事件条（观测/填充/SEC）+ 包信息 + 光变曲线（1ms bin）
- `scripts/gbm/plot_hxmt_vs_gbm.py` — 跨卫星验证：HXMT/HE 重建光变 vs Fermi/GBM
- `scripts/gecam/plot_hxmt_vs_gecam.py` — 跨卫星验证：HXMT/HE 重建光变 vs GECAM-C

`plot_fifo_resets.py` 用法：
```
python3 scripts/saturation/plot_fifo_resets.py --grb 2 --idx 1 40   # 只画第 1、40 个 reset
python3 scripts/saturation/plot_fifo_resets.py --grb 2               # 画全部
```

## 跨卫星验证

用独立卫星的未饱和光变曲线验证 HXMT/HE 的饱和重建结果。方法：扣本底后的净光变曲线对比，用斜线填充区分三种曲线（HXMT 观测 // C0、HXMT 填充 // C1、参考卫星 \\\\ C2）。

### 时间系统注意事项

不同卫星使用不同的 MET（Mission Elapsed Time）系统，转换到 UTC 时须注意：

**闰秒**：Python `datetime.total_seconds()` 不处理闰秒。HXMT MET epoch（2012-01-01）到 2026 年间有 3 个闰秒（2012-06-30、2015-06-30、2016-12-31），直接用 Python 会差 3 秒。必须用 astropy.time 做精确转换。

**FITS 时间标度**：FITS header 中 TIMESYS 关键字指定时间标度（UTC 或 TT）。MJDREFI + MJDREFF 定义参考时刻，须按 TIMESYS 指定的标度解析：
- 若 TIMESYS=TT：参考时刻在 TT 标度，TIME 列为 TT 秒
- 若 TIMESYS=UTC：参考时刻在 UTC 标度，TIME 列为 UTC 秒
- GECAM-C 的 MJDREFF=0.00080074074 天=69.184 秒，恰好是 TT-UTC 偏移量，表明 epoch 实际为 MJD 59215.0 UTC

**光行差**：两个 LEO 卫星最大距离 ~14000 km，光行差 <47ms，在 0.5s 以上的 bin 宽度下可忽略。

### GRB 260226A × Fermi/GBM

| 项目 | 值 |
|------|-----|
| GBM 触发号 | bn260226443 |
| GBM 数据 | `data/fermi_gbm/bn260226443/glg_tte_{n0,n3,b0}_*.fit` |
| 能量匹配 | NaI ch72-124（199-909 keV）+ BGO b0 ch0-19（113-930 keV）→ ~200-900 keV |
| 时间偏移 | GBM T=0 比 HXMT T=0 晚 5.958s |
| 本底区间 | [-10,-2]+[60,80]s |
| GBM 饱和 | 未饱和（GCN 43851 未提及饱和） |

GBM TTE 数据来源：`https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2026/bn260226443/current/`

```bash
# Paper Fig. 7 (f7_xsat_260226_gbm.pdf) standard command, 2026-07-12.
# Reproduces HXMT/GBM = 1.07±0.22 (38 bins), HXMT/eng = 1.05±0.38 (42 bins), GBM scale ×0.73.
# Scale fit excludes filler-containing bins (26 of 40 used) to avoid circularity:
# the normalization must not depend on the reconstruction under test.
# bkg windows avoid the no-data edge [-10,-6.5] of the --before 10 reconstruction window.
python3 scripts/gbm/plot_hxmt_vs_gbm.py --bin 0.5 --det n0 n3 b0 \
    --bkg -6.5 -2 60 80 --scale-range 20 40 --xlim -6.5 80 --pub \
    -o figures/f7_xsat_260226_gbm.pdf
# (legacy, pre-2026-07: --emin 200 --emax 900; energy filter since restored as optional)

# MERGED (2026-07-17): the paper now uses a single 4-panel figure in the
# former f7 slot (top: broadband; lower three: NaI bands), replacing the
# separate f7 + f14 pair:
python3 scripts/bursts/plot_260226_combo.py -o figures/f7_xsat_260226_gbm.pdf
# Reproduces broadband ×0.73 / 1.07±0.22 (38 bins) and NaI scales
# ×0.023/0.070/0.113; NaI filler-bin ratio medians 1.18/1.23/1.18
# (paper Sec. 6.5 & Table 4).
# 2026-07-17 FIX: NaI linear-background window moved from (-8,-3) to
# (-4.5,-0.5); the old left window fell in the dead zone before the
# first SEC anchor (NaI data starts ~-5s) and corrupted the linear fit
# (spuriously low soft-band medians 0.78/0.82). Corrected result: all
# three NaI bands sit ~20% high, same sign as the broadband, a
# rate-dependent cross-instrument offset (recovery validated by dug-gap
# + engineering channel).
# CAUTION: the filler energy assignment slices gaps into sub-windows
# from the reconstruction window start, so per-filler (channel, pw)
# depends on --before/--after. The published NaI numbers require the
# f14 window (--before 8 --after 80); plot_260226_combo.py hardcodes it.
# Former standalone NaI-bands command (reference only):
# python3 scripts/bursts/plot_hxmt_vs_gbm_bands.py -o f14_xsat_260226_bands.pdf

# MERGED (2026-07-18): the paper now uses one 4-panel figure at 0.1 s for
# 250919A (top: 70-700 keV broadband; lower three: 70-150/150-300/300-700
# keV bands), replacing the separate f15 (0.5 s bands) + f16 (0.1 s single
# band) pair:
python3 scripts/bursts/plot_250919_combo.py -o figures/f15_xsat_250919_bands.pdf
# Filler-bin HXMT/ext medians: 300-700 keV 1.12 (GBM)/1.08 (SVOM); rising
# to 1.53/1.40 at 70-150 keV; 1.51/1.37 broadband. Scale fit on the
# filler-free bins of T0-2..+15; linear background on -30..-24 & 112..130.
# Former standalone commands (reference only):
# python3 scripts/bursts/plot_hxmt_csi_multi.py --burst 250919A --bin 0.5 --bands \
#     --xlim -12 28 --pub -o f15_xsat_250919_bands.pdf
# python3 scripts/bursts/plot_hxmt_csi_multi.py --burst 250919A --bin 0.1 --band 300 700 \
#     --xlim 2 13 --pub -o f16_xsat_250919_100ms.pdf

# Paper Fig. 11 (f11_200415_failure.pdf): scripts/bursts/plot_200415_1ms_failure.py
# (no args; writes 200415A_hxmt_vs_asim_1ms.pdf, copy to figures/f11_200415_failure.pdf).
# Paper Fig. 4: scripts/grb221009a/plot_fig4_solve_vs_1k_221009a.py; Fig. 5:
# scripts/saturation/plot_fig3_wrap_uniqueness.py (both call the current blink CLI;
# leap-second-aware since d1a62f6, T0 = 13:17:00).
```

### GRB 221009A × GECAM-C

| 项目 | 值 |
|------|-----|
| GECAM-C 数据 | `data/gecam_c/gcg_evt_221009_13_v09.fits`（GRD_EVT，10 个探测器） |
| 增益选择 | GAIN_TYPE=1（低增益，未饱和） |
| GECAM 时间系统 | TIMESYS=TT，epoch = MJD 59215.00080074074 TT = 2021-01-01 00:00:00 UTC |
| GECAM GTI 覆盖 | T+397~676s（主脉冲期间被地球遮挡，仅覆盖尾部辐射） |
| 本底区间 | [620,670]s（GECAM 有数据的安静区间） |
| 缩放区间 | [400,670]s（两者都有数据的重叠区间） |
| 饱和 | GECAM-C 低增益未饱和（An et al. 2023） |

GECAM-C 数据来源：`guohx@lxlogin.ihep.ac.cn:/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily/2022/10/09/GRD_EVT/`

GRB 221009A 重建耗时较长（~12000 个 gap），建议先缓存再绘图：
```bash
# 缓存重建结果（需要几分钟）
HXMT_1B_DIR=data/1B HXMT_1K_DIR=data/1K ./target/release/blink_cli sat 2022-10-09T13 \
  reconstruct 2022-10-09T13:17:02 --before 50 --after 750 > data/cache_221009a_reconstruct.csv

# 秒出图
python3 scripts/gecam/plot_hxmt_vs_gecam.py --cache data/cache_221009a_reconstruct.csv \
  --bin 1.0 --bkg 620 670 620 670 --scale-range 400 670 --xlim 397 676
```

### GRB 260226A 各卫星观测汇总

| 卫星 | 观测 | 饱和 | 数据可用 | GCN |
|------|------|------|----------|-----|
| HXMT/HE | 是 | 是 | 本地 1B 数据 | 43865 |
| Fermi/GBM | 是 | 否 | HEASARC TTE | 43851 |
| Konus-Wind | 是 | 否 | 仅图片 | 43864 |
| CALET/CGBM | 是 | 否 | DARTS 未发布（延迟~1月） | 43860 |
| AstroSat CZTI | 是 | 是 | — | 43846 |
| SVOM/GRM | 否 | — | — | — |
| GECAM | 否 | — | — | — |

### GRB 221009A 各卫星观测汇总

| 卫星 | 观测 | 饱和 | 数据可用 | 备注 |
|------|------|------|----------|------|
| HXMT/HE | 是 | 是 | 本地 1B 数据 | — |
| GECAM-C | 是 | 否（低增益） | 服务器 HEBS L1 | 主脉冲期间被地球遮挡，仅 T+397~676s |
| Fermi/GBM | 是 | 是（严重） | — | 不可用 |
| Konus-Wind | 是 | 部分 | — | — |
| INTEGRAL | 是 | 是 | — | — |
| SIRI-2 | 是 | 否 | 未知 | 400 keV - 10 MeV |

## 待设计

1. FIFO 复位边界的精确检测与处理
2. Dead zone 事件的恢复（当前 ~4.6% 的事件因 ptime 落在 [500000, 524288) 区间而无法分配）
