#!/usr/bin/env bash
# Итерация 2: обойти BGE-M3 через hard-negatives + seq512 + низкий LR.
# Фокус на bge (его и надо обойти). Всегда возвращает сервисы (trap).
#   nohup bash scripts/run_iter2.sh > runs/iter2.log 2>&1 &
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

echo "[$(ts)] === build QA pairs ==="
python scripts/02_build_pairs.py --sources zerde kazqad_nq kazqad_rp

echo "[$(ts)] === mine hard-negatives (bge-m3 miner) ==="
python scripts/05_mine_negatives.py --model models/bge-m3-st \
    --in data/pairs/train.jsonl --out data/pairs/train_triplets.jsonl \
    --num-neg 4 --limit 40000 \
    || { echo "[$(ts)] MINE FAILED"; exit 1; }

echo "[$(ts)] === preflight bge (triplets, seq512) ==="
python scripts/04_train.py --base bge --max-steps 3 --batch 8 --mini-batch 8 \
    --loss cached --max-seq 512 --out runs/_pf2 \
    --train data/pairs/train_triplets.jsonl --eval /no \
    || { echo "[$(ts)] PREFLIGHT FAILED"; exit 1; }
rm -rf runs/_pf2
echo "[$(ts)] preflight OK"

echo "[$(ts)] === train bge heavy (hard-neg, seq512, lr1e-5) ==="
python scripts/04_train.py --base bge --epochs 1 --batch 32 --mini-batch 16 \
    --loss cached --max-seq 512 --lr 1e-5 --out runs/kaz-bge2 \
    --train data/pairs/train_triplets.jsonl --eval /no \
    || { echo "[$(ts)] TRAIN FAILED"; exit 1; }

echo "[$(ts)] === eval (bge-m3 vs kaz-bge2) ==="
python scripts/03_eval.py --models models/bge-m3-st runs/kaz-bge2/final --tasks kazqad ktd infoqa

echo "[$(ts)] === ITER2 DONE ==="
cat results/leaderboard.json
