#!/bin/bash
# TGF 闪电关联过滤 —— 在全部 search worker 跑完后单进程执行一次。
# 读取 data/Insight-HXMT_HE/**/*_signals.json 的全部候选，做 WWLLN 闪电关联，
# 原子写出 tgfs.json (temp + rename)。
# 提交： hep_sub -g hxmt -mem 8192 run_filter.sh

export PATH=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin:$PATH

# ── 需按实际部署确认 ──────────────────────────────────────────────
WORKDIR=/scratchfs/gecam/guohx/blink
export WWLLN_DB_PATH=/scratchfs/gecam/guohx/WWLLN/WWLLN.db   # 改成实际 WWLLN 库路径
# ─────────────────────────────────────────────────────────────────

cd "$WORKDIR" || exit 1
./blink wwlln
