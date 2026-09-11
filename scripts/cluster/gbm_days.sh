#!/bin/bash
# 列出可搜索的天。用法：gbm_days.sh <输出文件> [起始 YYYY-MM-DD] [结束 YYYY-MM-DD]
#
# 两个条件都要满足，缺一不可：
#   1. BGO 逐小时 TTE 齐全（48 个文件 = 2 探头 x 24 小时）。NaI 有没有由 Chunk
#      自己决定（2017-10 之前没有，那时按单组搜）。
#   2. 当天的 poshist 能找到。**这一条是后加的**：镜像有五种目录布局，
#      2014-12 与 2015-05..07 两段的 poshist 不在主目录下，那些天光有 TTE、
#      搜起来每小时都因 FileNotFound 被 missing_data 排除，账面搜了、实际零
#      曝光，而且不报错。布局清单见 gbm_layout_scan.sh 与 io/file.rs。
D=${FERMI_GBM_DIR:-/hxmtfs/data/Fermi_GBM}
out=${1:-/scratchfs2/gecam/guohx/gbmrun/days.txt}
from=${2:-2012-01-01}
to=${3:-2020-12-31}

# $1 = YYYY, $2 = MM, $3 = YYMMDD —— 与 io/file.rs 的 day_dirs() 一一对应
has_poshist() {
    ls "$D/$1/$2/${3:4:2}/current/glg_poshist_all_$3_v"* \
       "$D/$1/$3/tte/glg_poshist_all_$3_v"* \
       "$D/$1/$2/$3/tte/glg_poshist_all_$3_v"* \
       "$D/BGO/$1/$3/glg_poshist_all_$3_v"* \
       "$D/poshist/glg_poshist_all_$3_v"* 2>/dev/null | head -1
}

: > "$out"
no_bgo=0
no_pos=0
for y in $(seq "${from:0:4}" "${to:0:4}"); do
    for d in $(ls "$D/BGO/$y" 2>/dev/null); do
        day="20${d:0:2}-${d:2:2}-${d:4:2}"
        if [[ "$day" < "$from" ]] || [[ "$day" > "$to" ]]; then continue; fi
        n=$(ls "$D/BGO/$y/$d" 2>/dev/null | grep -cE 'glg_tte_b[01]_[0-9]{6}_[0-9]{2}z')
        if [ "$n" -ne 48 ]; then no_bgo=$((no_bgo + 1)); continue; fi
        if [ -z "$(has_poshist "$y" "${d:2:2}" "$d")" ]; then
            no_pos=$((no_pos + 1))
            echo "  无 poshist: $day" >&2
            continue
        fi
        echo "$day" >> "$out"
    done
done
sort -u "$out" -o "$out"
echo "可搜天数: $(wc -l < "$out")   （BGO 不齐 $no_bgo 天，poshist 缺 $no_pos 天）"
echo "首: $(head -1 "$out")   末: $(tail -1 "$out")"
