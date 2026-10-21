#!/bin/bash
# 从 lxlogin 拉取指定小时的 HXMT/HE 数据（1B sci + 1K Evt/Orbit/Att）
# 用法: ./scripts/saturation/pull_data.sh 2020-04-15T08

# set -e  # 不使用 set -e，单个文件失败不中断

if [ $# -ne 1 ]; then
    echo "Usage: $0 <YYYY-MM-DDTHH>"
    echo "Example: $0 2020-04-15T08"
    exit 1
fi

DATETIME="$1"
YEAR="${DATETIME:0:4}"
MONTH="${DATETIME:5:2}"
DAY="${DATETIME:8:2}"
HOUR="${DATETIME:11:2}"
YYYYMMDD="${YEAR}${MONTH}${DAY}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# 计算距离 2017-06-15 的天数 + 1（1K 文件路径需要）
NUM=$(python3 -c "from datetime import date; print((date(int('${YEAR}'),int('${MONTH}'),int('${DAY}')) - date(2017,6,15)).days + 1)")

echo "=== Pulling HXMT/HE data for ${YYYYMMDD}T${HOUR} ==="
echo "    1K num: ${NUM}"

# ─── 1B 数据（sci: A=0642, B=0922, C=1686; eng: A=0766, B=1009, C=1781）───
REMOTE_1B="/hxmtfs/data/Archive_tmp/1B/${YEAR}/${YYYYMMDD}"
LOCAL_1B="${BASE_DIR}/data/1B/${YEAR}/${YYYYMMDD}"

for CODE in 0642 0922 1686 0766 1009 1781; do
    LOCAL_DIR="${LOCAL_1B}/${CODE}"
    mkdir -p "$LOCAL_DIR"

    echo "--- 1B/${CODE} ---"
    REMOTE_FILE=$(ssh lxlogin "ls ${REMOTE_1B}/${CODE}/ | grep '${YYYYMMDD}T${HOUR}'" 2>/dev/null | tail -1)
    if [ -z "$REMOTE_FILE" ]; then
        echo "  WARNING: No file found for ${CODE}"
        continue
    fi

    if [ -f "${LOCAL_DIR}/${REMOTE_FILE}" ]; then
        echo "  Already exists: ${REMOTE_FILE}"
    else
        echo "  Downloading: ${REMOTE_FILE}"
        rsync -az --progress "lxlogin:${REMOTE_1B}/${CODE}/${REMOTE_FILE}" "${LOCAL_DIR}/"
    fi
done

# ─── 1K 数据（Evt, Orbit, Att）───
REMOTE_1K="/hxmt/work/HXMT-DATA/1K/Y${YEAR}${MONTH}/${YYYYMMDD}-$(printf '%04d' ${NUM})"
LOCAL_1K="${BASE_DIR}/data/1K/Y${YEAR}${MONTH}/${YYYYMMDD}-$(printf '%04d' ${NUM})"
mkdir -p "$LOCAL_1K"

for TYPE in "HE-Evt" "Orbit" "Att"; do
    echo "--- 1K/${TYPE} ---"
    REMOTE_FILE=$(ssh lxlogin "ls ${REMOTE_1K}/ | grep 'HXMT_${YYYYMMDD}T${HOUR}_${TYPE}_FFFFFF_V'" 2>/dev/null | sort | tail -1)
    if [ -z "$REMOTE_FILE" ]; then
        echo "  WARNING: No file found for ${TYPE}"
        continue
    fi

    if [ -f "${LOCAL_1K}/${REMOTE_FILE}" ]; then
        echo "  Already exists: ${REMOTE_FILE}"
    else
        echo "  Downloading: ${REMOTE_FILE}"
        rsync -az --progress "lxlogin:${REMOTE_1K}/${REMOTE_FILE}" "${LOCAL_1K}/"
    fi
done

echo ""
echo "=== Done ==="
echo "  1B dir: ${LOCAL_1B}"
echo "  1K dir: ${LOCAL_1K}"
