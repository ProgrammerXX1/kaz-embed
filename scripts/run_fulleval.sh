#!/usr/bin/env bash
# Честный замер на полном корпусе KazQAD (825k). Останавливает сервисы, вернёт по trap.
#   nohup bash scripts/run_fulleval.sh > runs/fulleval.log 2>&1 &
set -uo pipefail
PROJ=/root/kaz-embed; cd "$PROJ"
# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
ts(){ date +%H:%M:%S; }
restore(){ echo "[$(ts)] TRAP: возврат сервисов"; for s in llama-server ollama; do systemctl start "$s" && echo "  started $s"; done; }
trap restore EXIT INT TERM HUP

echo "[$(ts)] === освобождаю GPU ==="
for s in llama-server ollama; do
  systemctl is-active --quiet "$s" && systemctl stop "$s" && echo "  stopped $s" || true
done
sleep 6
nvidia-smi --query-gpu=memory.free --format=csv,noheader

echo "[$(ts)] === full-corpus eval: BGE-M3 vs kaz-bge2 (ключевая пара) ==="
python scripts/07_eval_fullcorpus.py --models \
    models/bge-m3-st runs/kaz-bge2/final \
    || { echo "[$(ts)] EVAL FAILED"; exit 1; }

echo "[$(ts)] === FULLEVAL DONE ==="
cat results/leaderboard_fullcorpus.json
