#!/usr/bin/env bash
# Sweep LoRA-конфигов на одном GPU. Каждый: train -> merge -> collapse-gate ->
# screening-eval на 150k пассажей (быстро). Результаты в results/sweep_150k.json.
# Конфиги из scripts/sweep_configs.txt: label rank alpha lr epochs maxsteps datafile
#   nohup bash scripts/run_sweep.sh > runs/sweep.log 2>&1 &
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

CFG="${1:-scripts/sweep_configs.txt}"
LIMIT="${LIMIT:-150000}"

echo "[$(ts)] === освобождаю GPU ==="
for s in llama-server ollama; do
  systemctl is-active --quiet "$s" && systemctl stop "$s" && echo "  stopped $s" || true
done
sleep 6

# бейзлайн bge-m3 на том же лимите — для сопоставимости (один раз)
if ! grep -q '"models/bge-m3-st"' results/sweep_150k.json 2>/dev/null; then
  echo "[$(ts)] === baseline bge-m3 @${LIMIT} ==="
  python scripts/07_eval_fullcorpus.py --models models/bge-m3-st --limit-corpus "$LIMIT" --out sweep_150k.json || true
fi

while read -r label rank alpha lr ep maxsteps data; do
  case "$label" in ''|'#'*) continue;; esac
  echo "[$(ts)] ===== SWEEP $label : r$rank a$alpha lr$lr ep$ep steps$maxsteps $data ====="
  ms=""; [ "${maxsteps:-0}" -gt 0 ] 2>/dev/null && ms="--max-steps $maxsteps"
  python scripts/04_train.py --base bge --lora "$rank" --lora-alpha "$alpha" --loss cached \
      --epochs "$ep" $ms --batch 64 --mini-batch 16 --max-seq 512 --lr "$lr" --grad-ckpt off \
      --out "runs/sw-$label-ad" --train "data/pairs/$data" --eval /no \
      || { echo "[$(ts)] $label TRAIN FAIL"; continue; }
  python scripts/08_merge_lora.py --base models/bge-m3-st \
      --adapter "runs/sw-$label-ad/final" --out "runs/sw-$label" \
      || { echo "[$(ts)] $label MERGE FAIL"; continue; }
  set +e
  python scripts/09_collapse_check.py --model "runs/sw-$label" --thresh 0.6; gate=$?
  set -e
  if [ $gate -ne 0 ]; then echo "[$(ts)] $label COLLAPSE (gate=$gate) — eval пропущен"; rm -rf "runs/sw-$label-ad"; continue; fi
  python scripts/07_eval_fullcorpus.py --models "runs/sw-$label" --limit-corpus "$LIMIT" --out sweep_150k.json \
      || echo "[$(ts)] $label EVAL FAIL"
  rm -rf "runs/sw-$label-ad"     # чистим адаптер (merged модель оставляем)
done < "$CFG"

echo "[$(ts)] === SWEEP DONE ==="
python3 -c "
import json
d=json.load(open('results/sweep_150k.json'))
print('=== SWEEP @150k (по nDCG@10) ===')
for k,v in sorted(d.items(),key=lambda x:-x[1].get('nDCG@10',0)):
    print(f\"{k:26s} nDCG@10={v.get('nDCG@10',0):.4f} MRR={v.get('MRR',0):.4f} R@1={v.get('Recall@1',0):.4f}\")
"
