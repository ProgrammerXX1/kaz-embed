#!/usr/bin/env bash
# Итерация 5 — против КОЛЛАПСА: щадящая LoRA (lr2e-5, alpha=rank, 2 негатива) +
# гейт по коллапсу перед дорогим full-eval. Всё то же (LoRA, full-corpus негативы,
# данные под задачу), но без пережатия, которое схлопывало пространство.
#   nohup bash scripts/run_iter5.sh > runs/iter5.log 2>&1 &
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

# триплеты из полного корпуса переиспользуем (созданы в iter4); если нет — майним
if [ ! -s data/pairs/train_triplets_full.jsonl ]; then
  python scripts/02_build_pairs.py --sources kazqad_nq kazqad_rp
  python scripts/05_mine_negatives.py --model models/bge-m3-st \
      --in data/pairs/train.jsonl --out data/pairs/train_triplets_full.jsonl --num-neg 5 --corpus-full
fi

echo "[$(ts)] === срезаю до 2 негативов (скорость) ==="
python3 -c "
import json
with open('data/pairs/train_triplets_full.jsonl') as f, open('data/pairs/train_triplets2.jsonl','w') as g:
    for line in f:
        d=json.loads(line)
        if d.get('negative_1') and d.get('negative_2'):
            g.write(json.dumps({'anchor':d['anchor'],'positive':d['positive'],'negative_1':d['negative_1'],'negative_2':d['negative_2']},ensure_ascii=False)+chr(10))
print('готово')
"

echo "[$(ts)] === preflight: щадящая LoRA (r16 alpha16 lr2e-5) ==="
python scripts/04_train.py --base bge --lora 16 --lora-alpha 16 --loss cached --max-steps 3 \
    --batch 8 --mini-batch 8 --max-seq 512 --grad-ckpt off --out runs/_pf5 \
    --train data/pairs/train_triplets2.jsonl --eval /no \
    || { echo "[$(ts)] PREFLIGHT FAILED"; exit 1; }
rm -rf runs/_pf5
echo "[$(ts)] preflight OK"

echo "[$(ts)] === train щадящая LoRA -> адаптер ==="
python scripts/04_train.py --base bge --lora 16 --lora-alpha 16 --loss cached --epochs 1 \
    --batch 64 --mini-batch 16 --max-seq 512 --lr 2e-5 --grad-ckpt off --out runs/kaz-bge5-adapter \
    --train data/pairs/train_triplets2.jsonl --eval /no \
    || { echo "[$(ts)] TRAIN FAILED"; exit 1; }

echo "[$(ts)] === слияние адаптера ==="
python scripts/08_merge_lora.py --base models/bge-m3-st \
    --adapter runs/kaz-bge5-adapter/final --out runs/kaz-bge5 \
    || { echo "[$(ts)] MERGE FAILED"; exit 1; }

echo "[$(ts)] === ГЕЙТ по коллапсу (перед дорогим full-eval) ==="
python scripts/09_collapse_check.py --model runs/kaz-bge5 --thresh 0.6
GATE=$?
if [ $GATE -ne 0 ]; then
  echo "[$(ts)] ⚠ коллапс не устранён — full-eval пропускаю, нужны ещё мягче гиперпараметры"
  echo "[$(ts)] === ITER5 DONE (collapse) ==="; exit 0
fi
echo "[$(ts)] гейт пройден — пространство здоровое"

echo "[$(ts)] === ГЛАВНЫЙ замер: full-corpus (kaz-bge5 vs BGE-M3) ==="
python scripts/07_eval_fullcorpus.py --models runs/kaz-bge5 || { echo "[$(ts)] FULLEVAL FAILED"; exit 1; }
echo "[$(ts)] === pool-замер ==="
python scripts/03_eval.py --models runs/kaz-bge5 --tasks kazqad infoqa || true

echo "[$(ts)] === ITER5 DONE ==="
echo '--- FULL CORPUS ---'; cat results/leaderboard_fullcorpus.json
