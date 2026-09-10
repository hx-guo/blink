#!/bin/bash
# 补下 IHEP 镜像缺的 GBM poshist。镜像里 2014 年的 poshist 放在
# {ROOT}/2014/{yymmdd}/tte/ 下，2014-12-10 起整段没有，导致那 22 天
# 每一小时都被 missing_data 排除、一个候选都出不来。
# 镜像目录只读，所以下到 gbmrun/gbm_extra/{YYYY}/{yymmdd}/tte/，
# 跑的时候用 FERMI_GBM_DIR 指过去（find 会退回原镜像找 TTE 是不行的，
# 所以这里同时把 BGO TTE 软链过来）。
set -u
BASE=/scratchfs2/gecam/guohx/gbmrun/gbm_extra
MIRROR=/hxmtfs/data/Fermi_GBM
URL=https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily
ok=0; bad=0
for day in "$@"; do
    y=${day:0:4}; m=${day:5:2}; d=${day:8:2}; ymd=${y:2:2}$m$d
    dir=$BASE/$y/$ymd/tte
    mkdir -p "$dir"
    got=""
    for v in v01 v00 v02; do
        f=glg_poshist_all_${ymd}_${v}.fit
        [ -s "$dir/$f" ] && { got=$f; break; }
        if curl -sSf --max-time 300 -o "$dir/$f.part" "$URL/$y/$m/$d/current/$f" 2>/dev/null; then
            mv -f "$dir/$f.part" "$dir/$f"; got=$f; break
        fi
        rm -f "$dir/$f.part"
    done
    if [ -n "$got" ]; then
        sz=$(stat -c %s "$dir/$got")
        echo "OK  $day  $got  $sz"
        ok=$((ok+1))
    else
        echo "FAIL $day"
        bad=$((bad+1))
    fi
done
# BGO 的逐小时 TTE 整年软链过来：FERMI_GBM_DIR 指向本目录时，
# io::file::day_dirs 的第三个候选目录 {ROOT}/BGO/{YYYY}/{YYMMDD} 就能一次找齐。
mkdir -p "$BASE/BGO"
for y in 2014 2019; do
    [ -d "$MIRROR/BGO/$y" ] && ln -sfn "$MIRROR/BGO/$y" "$BASE/BGO/$y"
done
echo "poshist 补齐 $ok 天，失败 $bad 天"
