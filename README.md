# kaz-embed — Kazakh text-embedding benchmark & fine-tuning study

A reproducible benchmark and fine-tuning pipeline for **Kazakh (kk) text-embedding /
retrieval models**, built around a hard question: *can a Kazakh-specialized encoder beat
the strong multilingual baseline **BAAI/bge-m3** on Kazakh retrieval?*

Everything here is **code + research notes + honest results**. The datasets, model weights
and large intermediate files are **not** in this repo (they live on the training box); see
`data/datasets.yaml` for exact Hugging Face sources.

## TL;DR (honest state)

- We evaluate on the **full KazQAD corpus** (825k passages, 1929 native test queries) — the
  real retrieval setting, not a small pool.
- **BAAI/bge-m3 is a very strong baseline on Kazakh.** Our best Kazakh-specialized model
  (`wise7-0.5`) reaches **statistical parity with it on the full corpus** — nominally ahead on
  MRR, within noise on nDCG@10/Recall@1 — and **beats bge-m3 at 150k-corpus scale**.
- Getting there took fixing three failure modes: naive full fine-tuning **regressed** bge-m3
  (catastrophic forgetting); aggressive LoRA **collapsed** the space (mean cosine 0.82 vs
  healthy 0.32); the winning recipe is **gentle LoRA (r16/α16/lr2e-5) + CachedGIST loss
  (false-negative masking) + hard negatives mined from the full corpus + WiSE-FT weight
  interpolation**.
- The gap to bge-m3 went from **−0.035 → −0.0017 nDCG@10** across 7 iterations.
- A clean, *significant* win likely needs the next lever: **in-domain synthetic doc2query**
  training data generated from the 825k corpus itself.

## Leaderboard — full corpus (825k passages)

| Model | nDCG@10 | MRR | Recall@1 | notes |
|---|---|---|---|---|
| BAAI/bge-m3 (baseline) | **0.3619** | 0.3485 | **0.2333** | zero-shot SOTA |
| **wise7-0.5** (ours, best) | 0.3602 | **0.3487** | 0.2317 | GIST + more data + WiSE-FT; **parity, MRR-ahead** |
| wise6-0.7 | 0.3584 | 0.3444 | 0.2255 | GIST + WiSE-FT |
| kaz-bge6 (GIST LoRA) | 0.3530 | 0.3390 | 0.2198 | GIST, α=1 |
| kaz-bge5 (gentle LoRA) | 0.3415 | 0.3285 | 0.2115 | no collapse |
| kaz-bge2 (full fine-tune) | 0.3268 | 0.3136 | 0.1965 | catastrophic forgetting |

At a 150k-passage corpus, our WiSE-FT models **beat** bge-m3 (e.g. 0.6672 vs 0.6640 nDCG@10).
Also fine-tuned **kaz-e5** which **beats its own base** `intfloat/multilingual-e5-large`.
Differences at the top are within noise (±0.002 over 1929 queries) — a defensible "beats"
claim needs paired-bootstrap significance over ≥3 seeds.

## What's in here

```
scripts/
  00_setup_env.sh          uv venv + torch cu12 + sentence-transformers/peft/FlagEmbedding/mteb
  01_download_data.py      download the sources in data/datasets.yaml
  02_build_pairs.py        assemble (anchor, positive) training pairs
  05_mine_negatives.py     hard-negative mining from the FULL 825k corpus (NV-Retriever style)
  06_score_teacher.py      cross-encoder (bge-reranker-v2-m3) teacher scoring for distillation
  04_train.py              LoRA / full FT; losses: MNRL, CachedMNRL, CachedGIST, MarginMSE; Matryoshka
  08_merge_lora.py         merge a LoRA adapter into the base -> standalone model
  10_wise_ft.py            WiSE-FT weight interpolation (theta = base + alpha*(ft-base))
  09_collapse_check.py     embedding-collapse gate (mean pairwise cosine)
  03_eval.py               eval harness: KazQAD (pool), KazakhTextDuplicates, informatics_kaz (GPU topk)
  07_eval_fullcorpus.py    full 825k-corpus KazQAD retrieval eval
  run_*.sh                 orchestrators for each experiment iteration
research/                  literature + dataset + methodology + evaluation research notes
data/datasets.yaml         manifest of all Hugging Face data sources (with licenses)
PLAN.md                    master plan, decisions and iteration log
```

## Key lessons (documented so others don't repeat them)

1. **Use LoRA, not full fine-tuning** — full FT of a strong base forgets broad retrieval.
2. **Watch for embedding collapse** — gate on mean pairwise cosine; aggressive lr/α collapse it.
3. **Mine hard negatives from the FULL corpus**, not just the positives — the model must learn
   to reject the real distractors it faces at eval.
4. **Evaluate on the full corpus** — small-pool "parity" is a reranking-like artifact.
5. **WiSE-FT** (scaling the merged delta by α<1) cheaply reverses forgetting.

## Reproduce

```bash
bash scripts/00_setup_env.sh                 # env on a CUDA box (RTX 3090 class)
huggingface-cli login                        # for gated datasets (KazQAD etc.)
python scripts/01_download_data.py --run --use eval pairs
python scripts/02_build_pairs.py --sources kazqad_nq kazqad_rp
python scripts/05_mine_negatives.py --corpus-full --num-neg 5
python scripts/04_train.py --base bge --lora 16 --lora-alpha 16 --loss gist --lr 2e-5
python scripts/08_merge_lora.py --adapter runs/kaz-bge/final --out runs/kaz-bge-merged
python scripts/07_eval_fullcorpus.py --models models/bge-m3-st runs/kaz-bge-merged
```

## Data & licenses

Training data is used under its original licenses (see `data/datasets.yaml`): KazQAD
(CC-BY-SA-4.0), Zerde-QA (ODC-BY), KazParC (CC-BY-4.0). Base models: bge-m3 (MIT),
multilingual-e5 (MIT); teacher bge-reranker-v2-m3 (Apache-2.0). This is a **research /
non-commercial** project; any released model weights inherit **CC-BY-SA-4.0** (KazQAD
ShareAlike). Credit to ISSAI and dataset authors.

## License

Code in this repository: **MIT** (see `LICENSE`). Data and any released model weights are
under their own licenses as noted above.
