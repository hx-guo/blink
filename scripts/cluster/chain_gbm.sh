#!/bin/bash
# GBM 全量跑的链条。用法：chain_gbm.sh <天表> <作业脚本> <worker 数> <内存 MB> [年份 ...]
#
# **队列空 ≠ 跑完。** 这是本脚本存在的唯一理由。作业可能被整批删掉（共用账号上
# 别人的 `hep_rm -a`）、可能被挤占、可能节点故障——每一种都不报错，队列就空了。
# 轮询到队列空就往下走，会在残缺的数据上跑出格式正确、数字看着合理的产物，
# **而且没有任何征兆**。所以这里不看队列，只看**应有天数齐没齐**，不齐就重投，
# 作业脚本对已完成的天是幂等的（跳过）。
#
# 跑完还要过 gbm_health.py 那道闸：列表天数 / 有曝光天数 / 被排除小时数三个数
# 对不上就非零退出，链条到此为止、不往下做统计。
set -u
R=/scratchfs2/gecam/guohx/gbmrun
DAYS=${1:?天表}
JOB=${2:?作业脚本}
WORKERS=${3:-20}
MEM=${4:-4000}
shift 4 || true
YEARS="$*"
MAX_ROUNDS=8
export PATH=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin:$PATH
cd "$R"
log=$R/farm_logs/chain_$(basename "$DAYS" .txt).log
say() { echo "$(date +%H:%M:%S) $*" >> "$log"; }

want=$(wc -l < "$DAYS")
done_days() {
    local n=0 d y m dd
    while read -r d; do
        y=${d:0:4}; m=${d:5:2}; dd=${d:8:2}
        [ -f "data/Fermi_GBM/$y/$m/$y$m${dd}_hours.json" ] \
            && [ -f "data/Fermi_GBM/$y/$m/$y$m${dd}_signals.json" ] && n=$((n + 1))
    done < "$DAYS"
    echo $n
}

say "chain 起：天表 $DAYS（$want 天），$WORKERS worker，-mem $MEM"
for round in $(seq 1 $MAX_ROUNDS); do
    have=$(done_days)
    if [ "$have" -ge "$want" ]; then break; fi
    # 共用账号上别人的作业也算并发，投之前先看总量（全队规矩：总并发 > 120 就等）
    while :; do
        total=$(condor_q -submitter guohx 2>/dev/null | tail -1 | awk '{print $1}')
        total=${total:-0}
        [ "$total" -le 120 ] && break
        say "账号总并发 $total > 120，等 5 分钟"
        sleep 300
    done
    # 队列里已经有自己的 worker 就别再投一批：同 idx 的两个 worker 会同时开同一天
    mine=$(hep_q -u guohx 2>/dev/null | grep -c "$(basename "$JOB")")
    if [ "$mine" -gt 0 ]; then
        say "第 $round 轮：已完成 $have/$want，队列里已有 $mine 个 worker，不重复投"
    else
        say "第 $round 轮：已完成 $have/$want，提交 $WORKERS 个 worker"
        hep_sub -g hxmt -mem "$MEM" "$JOB" -argu "%{ProcId}" -n "$WORKERS" >> "$log" 2>&1
    fi
    # 只要队列里还有自己的 worker 就等；队列空了不代表跑完，回到循环顶上重投
    while [ "$(hep_q -u guohx 2>/dev/null | grep -c "$(basename "$JOB")")" -gt 0 ]; do
        sleep 120
        say "  进行中 $(done_days)/$want"
    done
    say "第 $round 轮结束：$(done_days)/$want"
done

have=$(done_days)
if [ "$have" -lt "$want" ]; then
    say "ABORT：$MAX_ROUNDS 轮之后仍只有 $have/$want 天，不往下做统计"
    exit 1
fi
say "天数齐（$have/$want），跑体检"
python3 "$R/gbm_health.py" "$DAYS" "$R/data" $YEARS >> "$log" 2>&1
code=$?
say "体检退出码 $code"
[ $code -ne 0 ] && { say "ABORT：体检不过"; exit 1; }
say "DONE"
