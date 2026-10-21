#!/bin/bash

export PATH=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin:$PATH
cd /scratchfs/gecam/guohx/pipeline

# 检查上一次任务是不是还没计算完
if [ $(hep_q -u | wc -l) -ne 4 ]; then
  echo "Previous jobs are still running. Exiting."
  exit 0
fi

cp ./tgfs.json ../snapshot-stepping-visual/app/
pushd ../snapshot-stepping-visual
git add .
git commit -m "Auto update tgfs.json at $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main
popd

comm -23 \
    <(find . \( -name "pipeline_run.sh.out.*" -or -name "pipeline_run.sh.err.*" \) -and -size 0 | sort) \
    <(hep_q -u | grep pipeline_run.sh | awk '{ print "./pipeline_run.sh.out." $1 "\n./pipeline_run.sh.err." $1 }' | sort) \
    | xargs rm -rf

hep_sub -mem 8192 -g hxmt pipeline_run.sh
