#!/usr/bin/env bash
set -uo pipefail
PROJ=/root/kaz-embed; cd "$PROJ"; source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
ts(){ date +%H:%M:%S; }
restore(){ echo "[$(ts)] TRAP restore"; for s in llama-server ollama; do systemctl start "$s" && echo "  started $s"; done; }
trap restore EXIT INT TERM HUP
for s in llama-server ollama; do systemctl is-active --quiet "$s" && systemctl stop "$s" && echo "stopped $s" || true; done
sleep 6
echo "[$(ts)] === full-corpus kaz-bge5 (без e5-префиксов) ==="
python scripts/07_eval_fullcorpus.py --models runs/kaz-bge5 || { echo FULLEVAL FAILED; exit 1; }
echo "[$(ts)] === pool kaz-bge5 ==="
python scripts/03_eval.py --models runs/kaz-bge5 --tasks kazqad infoqa || true
echo "[$(ts)] === EVAL5 DONE ==="; cat results/leaderboard_fullcorpus.json
