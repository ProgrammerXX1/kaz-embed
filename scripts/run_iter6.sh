#!/usr/bin/env bash
# Итерация 6 — research-рецепт: gentle LoRA (r16/a16/lr2e-5) + CachedGISTEmbedLoss
# (guide=bge-m3 маскирует ложные негативы, temp0.03) на full-corpus-mined триплетах,
# затем WiSE-FT поверх. Цель — превзойти bge-m3 (0.362) чисто рецептом.
#   nohup bash scripts/run_iter6.sh > runs/iter6.log 2>&1 &
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

echo "[$(ts)] === preflight GIST LoRA ==="
python scripts/04_train.py --base bge --lora 16 --lora-alpha 16 --loss gist --temp 0.03 \
    --max-steps 3 --batch 8 --mini-batch 8 --max-seq 512 --grad-ckpt off --out runs/_pf6 \
    --train data/pairs/train_triplets2.jsonl --eval /no \
    || { echo "[$(ts)] PREFLIGHT FAILED"; exit 1; }
rm -rf runs/_pf6
echo "[$(ts)] preflight OK"

echo "[$(ts)] === train GIST LoRA -> адаптер ==="
python scripts/04_train.py --base bge --lora 16 --lora-alpha 16 --loss gist --temp 0.03 \
    --epochs 1 --batch 64 --mini-batch 16 --max-seq 512 --lr 2e-5 --grad-ckpt off \
    --out runs/kaz-bge6-ad --train data/pairs/train_triplets2.jsonl --eval /no \
    || { echo "[$(ts)] TRAIN FAILED"; exit 1; }

echo "[$(ts)] === слияние + гейт ==="
python scripts/08_merge_lora.py --base models/bge-m3-st --adapter runs/kaz-bge6-ad/final --out runs/kaz-bge6 \
    || { echo "[$(ts)] MERGE FAILED"; exit 1; }
set +e; python scripts/09_collapse_check.py --model runs/kaz-bge6 --thresh 0.6; gate=$?; set -e

echo "[$(ts)] === full-corpus kaz-bge6 + wise-0.3 (лучший WiSE-FT из kaz-bge5) ==="
python scripts/07_eval_fullcorpus.py --models runs/kaz-bge6 runs/wise-0.3 || echo "[$(ts)] eval FAIL"

echo "[$(ts)] === WiSE-FT поверх kaz-bge6 (screening @150k) ==="
for a in 0.3 0.5 0.7; do
  python scripts/10_wise_ft.py --base models/bge-m3-st --ft runs/kaz-bge6 --alpha $a --out runs/wise6-$a || continue
  python scripts/07_eval_fullcorpus.py --models runs/wise6-$a --limit-corpus 150000 --out sweep_150k.json || true
done

echo "[$(ts)] === ITER6 DONE ==="
python3 -c "
import json
for f in ['leaderboard_fullcorpus.json','sweep_150k.json']:
    try:
        d=json.load(open('results/'+f)); print('---',f,'---')
        for k,v in sorted(d.items(),key=lambda x:-x[1].get('nDCG@10',0)): print(f\"{k:24s} nDCG={v.get('nDCG@10',0):.4f} R@1={v.get('Recall@1',0):.4f}\")
    except Exception as e: print(f,e)
"
