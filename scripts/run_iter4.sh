#!/usr/bin/env bash
# Итерация 4 — ИСПРАВЛЕННЫЙ подход к обходу BGE-M3:
#  фикс1: LoRA (не ломаем базу, добавляем казахский)
#  фикс2: hard-negatives из ПОЛНОГО корпуса 825k (реальные дистракторы)
#  фикс3: данные под задачу — question->PASSAGE (KazQAD nq/rp), без коротких ответов Zerde
#   nohup bash scripts/run_iter4.sh > runs/iter4.log 2>&1 &
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

echo "[$(ts)] === фикс3: данные под задачу (question->passage) ==="
python scripts/02_build_pairs.py --sources kazqad_nq kazqad_rp

echo "[$(ts)] === фикс2: mine hard-neg из ПОЛНОГО корпуса 825k ==="
python scripts/05_mine_negatives.py --model models/bge-m3-st \
    --in data/pairs/train.jsonl --out data/pairs/train_triplets_full.jsonl \
    --num-neg 5 --corpus-full \
    || { echo "[$(ts)] MINE FAILED"; exit 1; }

echo "[$(ts)] === preflight: LoRA (3 шага) ==="
python scripts/04_train.py --base bge --lora 32 --loss cached --max-steps 3 \
    --batch 8 --mini-batch 8 --max-seq 512 --grad-ckpt off --out runs/_pf4 \
    --train data/pairs/train_triplets_full.jsonl --eval /no \
    || { echo "[$(ts)] PREFLIGHT FAILED"; exit 1; }
rm -rf runs/_pf4
echo "[$(ts)] preflight OK"

echo "[$(ts)] === фикс1: train LoRA поверх bge-m3 (1 эпоха, batch64, lr1e-4) ==="
python scripts/04_train.py --base bge --lora 32 --loss cached --epochs 1 \
    --batch 64 --mini-batch 16 --max-seq 512 --lr 1e-4 --grad-ckpt off --out runs/kaz-bge4 \
    --train data/pairs/train_triplets_full.jsonl --eval /no \
    || { echo "[$(ts)] TRAIN FAILED"; exit 1; }

echo "[$(ts)] === ГЛАВНЫЙ замер: full-corpus (kaz-bge4 vs сохранённый BGE-M3) ==="
python scripts/07_eval_fullcorpus.py --models runs/kaz-bge4/final \
    || { echo "[$(ts)] FULLEVAL FAILED"; exit 1; }

echo "[$(ts)] === контекст: pool-замер ==="
python scripts/03_eval.py --models runs/kaz-bge4/final --tasks kazqad infoqa || true

echo "[$(ts)] === ITER4 DONE ==="
echo '--- FULL CORPUS ---'; cat results/leaderboard_fullcorpus.json
