#!/bin/bash
# 提交 TGF 全量搜索：100 个 worker，%{ProcId}(0..99) 作为 worker 索引。
# 重跑安全：已处理的天会被 last_modified 检查跳过，所以中途失败/超时后
# 直接再跑一次本脚本即可补齐剩余的天。
# 注意：run_search.sh 里的 WORKERS 必须与这里的 -n 一致 (均为 100)。

export PATH=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin:$PATH

# HXMT 普通作业墙钟上限 14h；100 worker 时每作业 ~28 天 ≈ 2h，余量充足。
hep_sub -g hxmt -mem 8192 run_search.sh -argu "%{ProcId}" -n 100
