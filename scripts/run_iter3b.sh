#!/usr/bin/env bash
# Iter3b: только обучение+оценка (скоринг и distill-набор уже готовы из iter3).
# Дистилляция поверх kaz-bge2, ОГРАНИЧЕННАЯ по шагам (быстро). -> runs/kaz-bge3
#   nohup bash scripts/run_iter3b.sh >> runs/iter3.log 2>&1 &
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

echo "[$(ts)] === iter3b: освобождаю GPU ==="
for s in llama-server ollama; do
  systemctl is-active --quiet "$s" && systemctl stop "$s" && echo "  stopped $s" || true
done
sleep 6
nvidia-smi --query-gpu=memory.free --format=csv,noheader

[ -s data/pairs/train_distill.jsonl ] || { echo "[$(ts)] НЕТ distill-набора"; exit 1; }
echo "[$(ts)] distill rows: $(wc -l < data/pairs/train_distill.jsonl)"

echo "[$(ts)] === train kaz-bge3 (MarginMSE, ~1500 шагов, lr5e-6) ==="
python scripts/04_train.py --base bge --init runs/kaz-bge2/final --epochs 1 --max-steps 1500 \
    --batch 48 --loss marginmse --max-seq 512 --lr 5e-6 --out runs/kaz-bge3 \
    --train data/pairs/train_distill.jsonl --eval /no \
    || { echo "[$(ts)] TRAIN FAILED"; exit 1; }

echo "[$(ts)] === eval (bge-m3 vs kaz-bge2 vs kaz-bge3) ==="
python scripts/03_eval.py --models models/bge-m3-st runs/kaz-bge2/final runs/kaz-bge3/final \
    --tasks kazqad ktd infoqa

echo "[$(ts)] === ITER3 DONE ==="
cat results/leaderboard.json
