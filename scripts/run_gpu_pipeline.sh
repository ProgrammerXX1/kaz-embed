#!/usr/bin/env bash
# Автономная GPU-сессия v2: устойчива к падению отдельной базы.
# Освобождает 3090, per-base preflight (не фатальный), бейзлайны, обучает все
# прошедшие базы (bge->gte->e5), оценивает после каждой, ВСЕГДА возвращает сервисы.
#   nohup bash scripts/run_gpu_pipeline.sh > runs/pipeline.log 2>&1 &
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
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[$(ts)] === build pairs (QA-фокус: zerde+kazqad, без kazparc для скорости) ==="
python scripts/02_build_pairs.py --sources zerde kazqad_nq kazqad_rp

BASES_LIST="${BASES_LIST:-bge gte e5}"
echo "[$(ts)] === per-base preflight (3 шага, не фатально): $BASES_LIST ==="
PASS=""
for b in $BASES_LIST; do
  echo "[$(ts)] --- preflight $b ---"
  if CUDA_LAUNCH_BLOCKING=1 python scripts/04_train.py --base "$b" --max-steps 3 \
        --batch 16 --mini-batch 8 --out "runs/_pf_$b" \
        --train data/pairs/train.jsonl --eval /no/eval; then
    PASS="$PASS $b"; echo "[$(ts)] preflight $b OK"
  else
    echo "[$(ts)] preflight $b FAILED — пропуск"
  fi
  rm -rf "runs/_pf_$b"
done
echo "[$(ts)] preflight passed:$PASS"

# СНАЧАЛА обучение (ценное), оценку всех моделей делаем в конце одним быстрым проходом
TRAINED=""
for b in $PASS; do
  echo "[$(ts)] === train $b (fast: mnrl, seq256, batch64) ==="
  if python scripts/04_train.py --base "$b" --epochs 1 --batch 64 --max-seq 256 \
        --loss mnrl --matryoshka off \
        --train data/pairs/train.jsonl --eval data/pairs/dev.jsonl; then
    TRAINED="$TRAINED runs/kaz-$b/final"; echo "[$(ts)] trained kaz-$b"
  else
    echo "[$(ts)] train $b FAILED — дальше"
  fi
done

echo "[$(ts)] === ОЦЕНКА всех (бейзлайны + обученные), GPU-topk ==="
BGE_BASE="BAAI/bge-m3"; [ -d models/bge-m3-st ] && BGE_BASE="models/bge-m3-st"
python scripts/03_eval.py \
    --models "$BGE_BASE" intfloat/multilingual-e5-large sentence-transformers/LaBSE $TRAINED \
    --tasks kazqad ktd infoqa

echo "[$(ts)] === PIPELINE DONE ==="
cat results/leaderboard.json
