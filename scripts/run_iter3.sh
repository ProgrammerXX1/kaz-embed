#!/usr/bin/env bash
# Итерация 3: дистилляция из кросс-энкодера bge-reranker-v2-m3 ПОВЕРХ kaz-bge2
# (в котором уже запечён контрастив с hard-neg). Цель — решительно обойти BGE-M3.
#   nohup bash scripts/run_iter3.sh > runs/iter3.log 2>&1 &
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

# триплеты уже намайнены в iter2 (data/pairs/train_triplets.jsonl); если нет — намайним
if [ ! -s data/pairs/train_triplets.jsonl ]; then
  echo "[$(ts)] === (re)build pairs + mine ==="
  python scripts/02_build_pairs.py --sources zerde kazqad_nq kazqad_rp
  python scripts/05_mine_negatives.py --model models/bge-m3-st \
      --in data/pairs/train.jsonl --out data/pairs/train_triplets.jsonl --num-neg 4 --limit 40000
fi

echo "[$(ts)] === teacher scoring (bge-reranker-v2-m3) ==="
python scripts/06_score_teacher.py --in data/pairs/train_triplets.jsonl \
    --out data/pairs/train_distill.jsonl --reranker BAAI/bge-reranker-v2-m3 --max-neg 4 \
    || { echo "[$(ts)] TEACHER SCORING FAILED"; exit 1; }

echo "[$(ts)] === preflight distill (MarginMSE, из kaz-bge2) ==="
python scripts/04_train.py --base bge --init runs/kaz-bge2/final --max-steps 3 \
    --batch 16 --loss marginmse --max-seq 512 --out runs/_pf3 \
    --train data/pairs/train_distill.jsonl --eval /no \
    || { echo "[$(ts)] PREFLIGHT FAILED"; exit 1; }
rm -rf runs/_pf3
echo "[$(ts)] preflight OK"

echo "[$(ts)] === train distill: kaz-bge3 (MarginMSE, 2 эпохи, lr5e-6) ==="
python scripts/04_train.py --base bge --init runs/kaz-bge2/final --epochs 2 \
    --batch 48 --loss marginmse --max-seq 512 --lr 5e-6 --out runs/kaz-bge3 \
    --train data/pairs/train_distill.jsonl --eval /no \
    || { echo "[$(ts)] TRAIN FAILED"; exit 1; }

echo "[$(ts)] === eval (bge-m3 vs kaz-bge2 vs kaz-bge3) ==="
python scripts/03_eval.py --models models/bge-m3-st runs/kaz-bge2/final runs/kaz-bge3/final \
    --tasks kazqad ktd infoqa

echo "[$(ts)] === ITER3 DONE ==="
cat results/leaderboard.json
