#!/bin/bash
# 穷举 GBM 镜像里 poshist 与 TTE 的目录布局，别一种一种试。
#
# 前三种布局曾被当成全部，结果 2014-12 与 2015-05..07 两次出现「有 TTE 却找不到
# 星历、整段静默零曝光」。改动镜像覆盖范围（换年份、换任务）之前跑一次这个，
# 把输出的骨架数跟 `io/file.rs` 里 day_dirs() 的条目数对上。
D=${FERMI_GBM_DIR:-/hxmtfs/data/Fermi_GBM}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
find "$D" -maxdepth 5 -name 'glg_poshist_all_*' -printf '%h\n' 2>/dev/null | sort -u > "$tmp/pos.txt"
find "$D" -maxdepth 5 -name 'glg_tte_*'         -printf '%h\n' 2>/dev/null | sort -u > "$tmp/tte.txt"
D="$D" python3 - "$tmp/pos.txt" "$tmp/tte.txt" <<'PY'
import collections, os, re, sys
root = os.environ["D"].rstrip("/") + "/"
def skeleton(path):
    if path.startswith(root):
        path = path[len(root):]
    out = []
    for part in path.split("/"):
        if re.fullmatch(r"\d{4}", part):
            out.append("<YYYY>")
        elif re.fullmatch(r"\d{6}", part):
            out.append("<YYMMDD>")
        elif re.fullmatch(r"\d{2}", part):
            out.append("<NN>")
        else:
            out.append(part)
    return "/".join(out)
for label, path in (("poshist", sys.argv[1]), ("TTE", sys.argv[2])):
    counts, example = collections.Counter(), {}
    for line in open(path):
        key = skeleton(line.strip())
        counts[key] += 1
        example.setdefault(key, line.strip())
    print(f"=== {label}：{len(counts)} 种骨架，{sum(counts.values())} 个目录 ===")
    for key, n in counts.most_common():
        print(f"{n:7d}  {key:34s}  例 {example[key]}")
    print()
PY
