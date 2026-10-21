#!/bin/bash
# TGF 全量搜索 —— 单个 worker。
# 由 submit_search.sh 通过 hep_sub -n N 批量提交，%{ProcId}(0..N-1) 作为 worker 索引传入 $1。
# 每个 worker 按天 round-robin 处理自己的分片，结果原子写入
#   data/Insight-HXMT_HE/YYYY/MM/YYYYMMDD_signals.json   (temp + rename)
# 并按源文件 last_modified 跳过已处理的天 —— 可安全并行、断点重跑。

export PATH=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin:$PATH

# ── 需按实际部署确认 ──────────────────────────────────────────────
WORKDIR=/scratchfs/gecam/guohx/blink   # 含 blink 可执行文件 + data/ 输出目录
WORKERS=100                            # 必须与 submit_search.sh 的 -n 一致
FROM=2017-06-15                        # HXMT 发射日 (launch_day)
TO=2026-06-30                          # 任务终点 (重跑当天)
# 档案路径默认值与 IHEP 一致 (1B=/hxmtfs/data/Archive_tmp/1B)，如不同在此覆盖：
# export HXMT_1B_DIR=/hxmtfs/data/Archive_tmp/1B
# export HXMT_1K_DIR=/hxmt/work/HXMT-DATA/1K
# ─────────────────────────────────────────────────────────────────

WORKER=$1
cd "$WORKDIR" || exit 1
./blink search "$FROM" "$TO" --workers "$WORKERS" --worker "$WORKER"
