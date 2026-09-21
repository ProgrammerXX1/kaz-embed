# kaz-embed — Kazakh text-embedding benchmark & fine-tuning study

A reproducible benchmark and fine-tuning pipeline for **Kazakh (kk) text-embedding /
retrieval models**, built around a hard question: *can a Kazakh-specialized encoder beat
the strong multilingual baseline **BAAI/bge-m3** on Kazakh retrieval?*

Everything here is **code + research notes + honest results**. The datasets, model weights
and large intermediate files are **not** in this repo (they live on the training box); see
`data/datasets.yaml` for exact Hugging Face sources.

## TL;DR (current honest state)

- We evaluate on the **full KazQAD corpus** (825k passages, 1929 native test queries) — the
  real retrieval setting, not a small pool.
- **BAAI/bge-m3 is a very strong baseline on Kazakh** and hard to beat with a quick fine-tune.
- Naive full fine-tuning **regressed** it (catastrophic forgetting). Aggressive LoRA
  **collapsed** the embedding space (mean pairwise cosine 0.82 vs healthy 0.32).
- A **gentle LoRA** (r16, α16, lr2e-5) fixes collapse and gets closest so far; **WiSE-FT**
  weight interpolation squeaks marginally above baseline on a screening corpus.
- Work in progress: **CachedGIST loss + WiSE-FT + in-domain synthetic queries** to get a
  clean, significant win. Results below are updated as runs finish.

## Leaderboard — full corpus (825k passages), nDCG@10

| Model | nDCG@10 | MRR | Recall@1 | notes |
|---|---|---|---|---|
| BAAI/bge-m3 (baseline) | **0.362** | 0.348 | 0.233 | zero-shot |
| kaz-bge5 (gentle LoRA) | 0.3415 | 0.329 | 0.212 | no collapse; best fine-tune so far |
| kaz-bge2 (full fine-tune) | 0.327 | 0.314 | 0.196 | catastrophic forgetting |

Also fine-tuned **kaz-e5** which **beats its own base** `intfloat/multilingual-e5-large`
(a weaker base) on Kazakh — fine-tuning helps a weak base, but bge-m3 is a higher bar.

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
