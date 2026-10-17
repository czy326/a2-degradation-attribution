#!/usr/bin/env bash
# 图 1（引言：退化分布失配）一键重跑。
#   gen_samples 需要 GPU（BSRGAN 上采样）；已有 _cache 时自动跳过。
#   用法（Windows 侧）：wsl.exe -e bash -lc 'bash /mnt/e/新方向研究/a2/figures/run_all.sh'
set -u
cd "$(dirname "$0")" || exit 1
PY=/home/czy/envs/sr4sod/bin/python

if [ -f _cache/meta.json ]; then
  echo "[skip] 复用 _cache（删掉 _cache/meta.json 可强制重建）"
else
  echo "[gen] 生成样本（需 GPU）"
  "$PY" -u gen_samples.py || exit 1
fi

echo "[fig] 出图"
"$PY" -u make_fig01_mismatch.py 2>&1 | tee fig.log

echo "[fig2-7] 统计型配图"
"$PY" -u make_figs_a2.py 2>&1 | tee figs_a2.log

echo "[audit]"
grep -E "no text overlap|layout warning|TEXT/" fig.log || echo "  (无 audit 输出)"
ls -l out/ | sed 's/  */ /g'
