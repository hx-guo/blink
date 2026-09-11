#!/bin/bash
# GECAM-C 特征提取的按天分片。单线程一天约 12 分钟，30 天 6 小时，必须分片。
# 用法: hep_sub -g gecam job_gecam_feat.sh -argu "%{ProcId}" -n 10
# 任务清单 feat_tasks.txt 每行一个日期 yyyy-mm-dd；worker idx 取第 idx, idx+N, ... 天。
#
# 换事例口径（比如双增益合并）之后必须先清空 feat/，否则下面这个「已产出就跳过」
# 会把旧口径的表当成已完成，新旧两种口径混进同一批结果。
set -u
cd /scratchfs2/gecam/guohx/gecamrun || exit 1
idx=${1:-0}
workers=${WORKERS:-10}
mkdir -p feat farm_logs
n=0
while read -r day; do
  [ -z "$day" ] && continue
  if [ $((n % workers)) -eq "$idx" ]; then
    stamp=${day//-/}
    yyyy=${day:0:4}; mm=${day:5:2}
    sig="data/GECAM-C/$yyyy/$mm/${stamp}_signals.json"
    out="feat/feat_${stamp}.csv"
    if [ -f "$sig" ] && [ ! -s "$out" ]; then
      python3 /scratchfs2/gecam/guohx/gecam_features.py "$sig" "$out" GECAM-C >> "farm_logs/feat_$idx.log" 2>&1
    fi
  fi
  n=$((n + 1))
done < feat_tasks.txt
echo "worker $idx 完成"
