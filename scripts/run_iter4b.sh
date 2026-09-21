#!/usr/bin/env bash
# Iter4b: спасаем обученный LoRA-адаптер kaz-bge4 (обучение было отличным, сломалось
# только сохранение). Сливаем адаптер с базой -> переоцениваем. БЕЗ переобучения.
#   nohup bash scripts/run_iter4b.sh > runs/iter4b.log 2>&1 &
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

echo "[$(ts)] === слияние LoRA-адаптера с bge-m3 ==="
python scripts/08_merge_lora.py --base models/bge-m3-st \
    --adapter runs/kaz-bge4/final --out runs/kaz-bge4-merged \
    || { echo "[$(ts)] MERGE FAILED"; exit 1; }

echo "[$(ts)] === ГЛАВНЫЙ замер: full-corpus (kaz-bge4-merged vs BGE-M3) ==="
python scripts/07_eval_fullcorpus.py --models runs/kaz-bge4-merged \
    || { echo "[$(ts)] FULLEVAL FAILED"; exit 1; }

echo "[$(ts)] === pool-замер (контекст) ==="
python scripts/03_eval.py --models runs/kaz-bge4-merged --tasks kazqad infoqa || true

echo "[$(ts)] === ITER4B DONE ==="
echo '--- FULL CORPUS ---'; cat results/leaderboard_fullcorpus.json