#!/bin/bash
# 重跑 2014-12-10..31：镜像缺 poshist，头一轮这 22 天 24/24 小时全被
# missing_data 排除、一个候选都没出。poshist 已由 gbm_get_poshist.sh 从
# HEASARC 补到 gbm_extra/，FERMI_GBM_DIR 指过去即可一次找齐（BGO 的逐小时
# TTE 由 gbm_extra/BGO/{YYYY} 软链回镜像）。
#
# 第 $1 个 worker：处理 days_dec2014.txt 中 (行号-1) % WORKERS == $1 的那些天。
cd /scratchfs2/gecam/guohx/gbmrun
export FERMI_GBM_DIR=/scratchfs2/gecam/guohx/gbmrun/gbm_extra
WORKERS=10
idx=$1
n=0
while read -r day; do
    if [ $((n % WORKERS)) -eq "$idx" ]; then
        ./blink search "$day" "$day" --instrument fermi-gbm
    fi
    n=$((n + 1))
done < days_dec2014.txt
