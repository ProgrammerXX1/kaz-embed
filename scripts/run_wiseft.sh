#!/usr/bin/env bash
# WiSE-FT α-sweep поверх дообученной модели (по умолчанию kaz-bge5). Дёшево (без
# обучения): интерполяция весов + screening-eval на 150k. Победителя — на полный корпус.
#   FT=runs/kaz-bge5 nohup bash scripts/run_wiseft.sh > runs/wiseft.log 2>&1 &
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

FT="${FT:-runs/kaz-bge5}"
ALPHAS="${ALPHAS:-0.2 0.3 0.4 0.5 0.7}"
LIMIT="${LIMIT:-150000}"

echo "[$(ts)] === освобождаю GPU ==="
for s in llama-server ollama; do
  systemctl is-active --quiet "$s" && systemctl stop "$s" && echo "  stopped $s" || true
done
sleep 6

# baseline bge-m3 @150k для сопоставимости
if ! grep -q '"models/bge-m3-st"' results/sweep_150k.json 2>/dev/null; then
  python scripts/07_eval_fullcorpus.py --models models/bge-m3-st --limit-corpus "$LIMIT" --out sweep_150k.json || true
fi
# сам FT (α=1) для сравнения
python scripts/07_eval_fullcorpus.py --models "$FT" --limit-corpus "$LIMIT" --out sweep_150k.json || true

for a in $ALPHAS; do
  out="runs/wise-$a"
  echo "[$(ts)] ===== WiSE-FT alpha=$a ====="
  python scripts/10_wise_ft.py --base models/bge-m3-st --ft "$FT" --alpha "$a" --out "$out" \
      || { echo "[$(ts)] wise $a FAIL"; continue; }
  python scripts/07_eval_fullcorpus.py --models "$out" --limit-corpus "$LIMIT" --out sweep_150k.json \
      || echo "[$(ts)] eval wise $a FAIL"
done

echo "[$(ts)] === WISEFT DONE ==="
python3 -c "
import json
d=json.load(open('results/sweep_150k.json'))
print('=== @150k (по nDCG@10) ===')
for k,v in sorted(d.items(),key=lambda x:-x[1].get('nDCG@10',0)):
    print(f\"{k:24s} nDCG@10={v.get('nDCG@10',0):.4f} MRR={v.get('MRR',0):.4f} R@1={v.get('Recall@1',0):.4f}\")
"
