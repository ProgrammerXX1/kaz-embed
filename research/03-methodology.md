# 03 — SOTA Methodology & Training Recipe

**Goal:** Train / fine-tune a Kazakh (`kk`) text **embedding** model that **beats BGE-M3 on Kazakh benchmarks**, on a **single RTX 3090 (24 GB)**.
**Date:** 2026-09-20 · **Scope:** dense retrieval embeddings (bi-encoder). Reranking is mentioned only as a *teacher*.

---

## 0. TL;DR (the recipe in 8 steps)

1. **Pick a base that is already decent at Kazakh** and fits 24 GB: fine-tune **`intfloat/multilingual-e5-large`** or **`BAAI/bge-m3`** itself (Track A, fastest win), or build a fresh modern encoder from **`jhu-clsp/mmBERT-base`** / **`Alibaba-NLP/gte-multilingual-base`** (Track B).
2. **Assemble Kazakh training pairs** from four sources: (a) native Kazakh QA/IR (KazQAD), (b) **machine-translated** EN retrieval + NLI sets (MS MARCO, NQ, HotpotQA, NLI), (c) **kk↔ru / kk↔en parallel** bitext, (d) **LLM-generated synthetic** query→passage pairs from a Kazakh corpus. Filter aggressively (LaBSE cosine + round-trip).
3. *(Optional, cheap)* **Continued pretraining** (MLM or RetroMAE) on a Kazakh mono-corpus if the base is weak on Kazakh.
4. **Mine hard negatives** with a first-pass retriever (BGE-M3 / e5) using **NV-Retriever settings** (`sampling_strategy="top"`, `relative_margin=0.05`, `max_score≈0.8`, 5–15 negs) to avoid false negatives.
5. *(Big lever)* **Distill from a strong teacher**: score (query, pos, negs) with the cross-encoder **`BAAI/bge-reranker-v2-m3`** and train with **listwise/MarginMSE distillation + contrastive** loss (or FlagEmbedding `m3_kd_loss`).
6. **Contrastive fine-tune** with **`CachedMultipleNegativesRankingLoss`** (GradCache → large effective batch on 24 GB) or **`CachedGISTEmbedLoss`** (guided false-negative filtering), wrapped in **`MatryoshkaLoss`** for flexible dims. bf16 + gradient checkpointing.
7. **Evaluate on Kazakh** the whole way: KazQAD-retrieval (nDCG@10 / Recall / MRR) + MMTEB Kazakh subset (Belebele, MASSIVE, SIB200, bitext), always **head-to-head vs BGE-M3**.
8. **Iterate**: re-mine hard negatives with *your improved* model (one or two rounds), re-distill, ship the checkpoint that wins on the held-out Kazakh eval.

**Biggest levers to actually beat BGE-M3 on Kazakh** (ranked): (1) high-quality **hard negatives with false-negative filtering**, (2) **volume of clean translated supervised triplets**, (3) **cross-encoder distillation**, (4) **LLM synthetic queries** on the target corpus, (5) **cross-lingual parallel anchoring**, (6) MLM/RetroMAE continued-pretraining. Data quality and distillation beat clever architecture.

---

## 1. What "beat BGE-M3 on Kazakh" means — target & evaluation

BGE-M3 (`BAAI/bge-m3`) is the baseline to beat: an **XLM-RoBERTa-large** backbone, **568 M params**, **1024-dim** dense output, **8192-token** context, trained with dense + sparse + ColBERT multi-vector heads and **self-knowledge-distillation**. It supports 100+ languages including Kazakh, but Kazakh is under-represented in its training mix — that gap is the opening.

**Evaluate continuously (not just at the end), always side-by-side vs BGE-M3.** Use the `mteb` library plus a custom KazQAD harness.

Primary (retrieval — the thing we optimize):
- **KazQAD-retrieval** — `issai/kazqad-retrieval` (HF). ~6k native questions (some from the Kazakh UNT exam), ~800k Kazakh Wikipedia passages, ~12k relevance judgements. Report **nDCG@10, MRR@10, Recall@100**. This is the headline number.
- **Belebele (kaz_Cyrl) Retrieval** (in MMTEB).
- **MultiWikiQA (kk)** reading-comprehension retrieval (in MMTEB).

Secondary (generalization / prove no regression):
- **MASSIVE** intent & scenario classification (kk), **SIB200** topic classification (kk), **KazSAnDRA** sentiment.
- **Bitext mining**: Tatoeba / FLORES **kk↔en, kk↔ru** (validates cross-lingual alignment).

Rule of thumb: run the *identical* MTEB task list on both your model and `bge-m3`; you must win the retrieval tasks and not lose the rest.

```bash
pip install mteb sentence-transformers
```
```python
import mteb
tasks = mteb.get_tasks(languages=["kaz"])          # all Kazakh MMTEB tasks
model = mteb.get_model("your-org/kaz-embed-v1")     # or a local path
mteb.MTEB(tasks=tasks).run(model, output_folder="results/kaz-embed-v1")
# repeat with mteb.get_model("BAAI/bge-m3") for the baseline, then diff.
```

---

## 2. Base model choice (what fits 24 GB and starts strong on Kazakh)

All candidates below are ≤ ~570 M params and **full-fine-tune comfortably on a single 3090** (see §5). Two tracks:

**Track A — specialize an existing multilingual *embedder* (fastest path to beating BGE-M3):**

| Model | Params | Ctx | Notes |
|---|---|---|---|
| `intfloat/multilingual-e5-large` | 560 M | 512 | Strong multilingual retriever, mean pooling, needs `query:`/`passage:` prefixes. Excellent starting point. |
| `BAAI/bge-m3` | 568 M | 8192 | Fine-tune the baseline *itself* on Kazakh — near-guaranteed to beat stock BGE-M3 on Kazakh. Safe, high-floor option. |
| `Alibaba-NLP/gte-multilingual-base` | ~305 M | 8192 | Modern, fast, long-context, strong retrieval, permissive. Great compute/quality tradeoff on a 3090. |
| `intfloat/multilingual-e5-base` | 278 M | 512 | Smaller/faster; bigger batches; good if you want fast iteration. |

**Track B — build a fresh encoder from a modern MLM backbone (highest ceiling, more work):**

| Model | Params | Ctx | Notes |
|---|---|---|---|
| `jhu-clsp/mmBERT-base` | ~307 M | 8192 | 2025 ModernBERT-style encoder, 3T tokens, **annealed low-resource curriculum** (explicitly boosts low-resource langs incl. Kazakh). Not an embedder yet → needs full contrastive pipeline. Very strong ceiling. |
| `nur-dev/kaz-roberta` (Kaz-RoBERTa) | base | 512 | Monolingual Kazakh MLM; best pure-Kazakh token modeling but no cross-lingual transfer and no embedding head. |
| `sentence-transformers/LaBSE` | 471 M | 512 | Great for bitext/alignment, weaker as a pure retriever. Use as a *filter*, not the main model. |

**Recommendation:**
- **Default / lowest-risk:** Track A on **`multilingual-e5-large`** (or `bge-m3` for the highest floor).
- **If you have time for the full pipeline and want the best result:** Track B on **`mmBERT-base`** (best modern low-resource backbone) with the complete data + distillation recipe below.
- Consider training **two** and keeping the winner — both fit the same 3090 recipe.

---

## 3. Objective & losses (contrastive learning for embeddings)

### 3.1 The core objective: InfoNCE / in-batch negatives
The workhorse is **`MultipleNegativesRankingLoss` (MNRL)** = InfoNCE = in-batch-negatives contrastive loss. Given a batch of (anchor, positive) pairs (optionally with explicit hard negatives), it maximizes similarity of each anchor to its positive and minimizes similarity to *all other positives/negatives in the batch*.

- **Temperature** matters a lot. MNRL uses `scale = 1/temperature` on cosine similarity; default `scale=20.0` (temp = 0.05). Retrieval SOTA typically uses **temp 0.02–0.05** (`scale` 20–50). Lower temp = sharper contrast, needs cleaner negatives.
- Use the **`NO_DUPLICATES`** batch sampler so a batch never contains two copies of the same text (which would create false in-batch negatives).
- **Larger batch = more in-batch negatives = better retriever.** This is why the cached variant below is central on a 24 GB card.

### 3.2 Large effective batch on 24 GB: CachedMNRL / GradCache
**`CachedMultipleNegativesRankingLoss` (CMNRL)** implements **GradCache**: it splits the batch into `mini_batch_size` chunks, does a no-grad forward to collect all embeddings/logits, computes the loss and caches per-example gradients, then re-runs a grad forward chunk-by-chunk to connect cached grads into the backward graph. Result: **effective batch size decoupled from memory** — you can run effective batches of 512–2048 on a 3090 while only ever holding `mini_batch_size` examples' activations. Cost: ~2–2.4× slower than plain MNRL, but consistently better models. Set `mini_batch_size` to the largest that fits (e.g. 8–32 at seq 512).

### 3.3 Hard negatives (the single biggest data lever)
In-batch negatives are "random"; **mined hard negatives** are passages that look relevant but aren't — they force the model to learn fine distinctions.

**How to mine:** use a first-pass retriever (BGE-M3, e5, or your own previous checkpoint) to retrieve top-k for each query, then select negatives from a rank window, **filtering out likely false negatives**. Follow **NV-Retriever** (arXiv 2407.15831) best settings:
- `sampling_strategy="top"`, `num_negatives` = 5–15 (≤10 recommended),
- `relative_margin=0.05` (a negative must be ≤ 95% as similar as the positive → positive-aware filtering that removes false negatives),
- `range_min≈10` (skip the very top ranks, which are often unlabeled positives), `max_score≈0.8` (cap absolute similarity), `use_faiss=True`.

Sentence Transformers utility:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from datasets import load_dataset

miner = SentenceTransformer("intfloat/multilingual-e5-large")  # or bge-m3, or your prev ckpt
dataset = load_dataset("json", data_files="kk_pairs.jsonl", split="train")  # cols: anchor, positive

hn = mine_hard_negatives(
    dataset=dataset, model=miner,
    anchor_column_name="anchor", positive_column_name="positive",
    range_min=10, range_max=100,
    max_score=0.8, relative_margin=0.05,
    num_negatives=10, sampling_strategy="top",
    output_format="n-tuple",   # (anchor, positive, neg_1..neg_n)
    use_faiss=True, batch_size=256,
)
hn.to_json("kk_triplets_minedHN.jsonl")
```
**Re-mine with your improved model** after the first training round (iterative hard-negative mining) — this is a reliable extra win.

### 3.4 Guided negatives: GISTEmbed (false-negative safety net)
Translated/synthetic Kazakh data is noisy, so false negatives are a real risk. **`GISTEmbedLoss` / `CachedGISTEmbedLoss`** use a **guide model** (e.g. `BAAI/bge-m3`) to *drop* in-batch candidate negatives that the guide judges too similar to the anchor/positive — yielding a cleaner, stronger training signal than plain MNRL. `CachedGISTEmbedLoss` combines this with GradCache for large batches. Recommended as the main loss when training on translated/synthetic data.
```python
from sentence_transformers.losses import CachedGISTEmbedLoss
guide = SentenceTransformer("BAAI/bge-m3")
loss = CachedGISTEmbedLoss(model, guide=guide, mini_batch_size=16, temperature=0.02)
```

### 3.5 Knowledge distillation from a strong teacher (big quality lever)
Pure contrastive fine-tuning of an already-strong model can *plateau or regress*; **adding distillation from a cross-encoder teacher gives consistent gains** (recent listwise-distillation work). Two practical routes:

**(a) FlagEmbedding native (`m3_kd_loss`)** — put teacher scores in the data (`pos_scores`, `neg_scores`) and set `--knowledge_distillation True`. For BGE-M3, `--use_self_distill True` also distills its own dense/sparse/colbert heads into each other.

**(b) Sentence-Transformers MarginMSE / listwise** — score every (query, passage) with the cross-encoder teacher, then train the bi-encoder to match teacher margins.
```python
from sentence_transformers.cross_encoder import CrossEncoder
teacher = CrossEncoder("BAAI/bge-reranker-v2-m3")  # multilingual, covers Kazakh
scores = teacher.predict([(q, p) for q, p in query_passage_pairs])  # cache to disk
# Train student with sentence_transformers.losses.MarginMSELoss (pos_score - neg_score targets)
# or a listwise KL/DistillKL objective alongside a small contrastive term.
```
Best practice: **distillation loss + a contrastive term together** (not distillation alone). The reranker sees full cross-attention over Kazakh text, so it transfers relevance knowledge the bi-encoder can't learn from labels alone.

### 3.6 Matryoshka representation learning (flexible dims, ~free)
Wrap the base loss in **`MatryoshkaLoss`** so truncated prefixes of the embedding (e.g. 256-d, 128-d) stay high-quality — enabling cheaper storage/search with minimal loss. It applies the loss to full and truncated dims and sums them; costs almost nothing and is standard in 2024–2026 SOTA embedders.
```python
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss
base = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16, scale=50.0)  # temp 0.02
loss = MatryoshkaLoss(model, base, matryoshka_dims=[1024, 768, 512, 256, 128, 64])
```

### 3.7 Loss decision table
| Situation | Recommended loss |
|---|---|
| Clean (anchor, positive) pairs, want max batch | `CachedMultipleNegativesRankingLoss` |
| Noisy translated/synthetic data (false-neg risk) | `CachedGISTEmbedLoss` (guide = bge-m3) |
| You have mined hard negatives | above losses with `n-tuple` (anchor, pos, neg_1..n) |
| You have teacher scores | FlagEmbedding `m3_kd_loss`, or ST `MarginMSELoss` + small contrastive |
| Any of the above + flexible dims | wrap in `MatryoshkaLoss` |

---

## 4. Low-resource-language specialization strategy

Ordered by expected win-per-effort on a low-resource language:

### 4.1 (Biggest) High-quality hard negatives + false-negative filtering
See §3.3–3.4. On low-resource data this is the highest-leverage step because labels are scarce and noisy; good negatives + GISTEmbed filtering extract maximum signal from limited pairs.

### 4.2 Translate English retrieval / NLI sets into Kazakh (volume of clean supervision)
The fastest way to get *thousands* of clean supervised triplets in Kazakh:
- **Sources:** MS MARCO, Natural Questions, HotpotQA, MrTyDi (retrieval); SNLI/MNLI/AllNLI (entailment → strong for embeddings); MTEB training mixtures.
- **MT engine:** **NLLB-200** (`facebook/nllb-200-3.3B` / distilled-1.3B, has `kaz_Cyrl`) locally, or a commercial API for quality. Translate queries and passages.
- **Quality filtering (critical):** keep a pair only if (i) **LaBSE cosine** between source and Kazakh translation > ~0.7 (adequate translation), and (ii) optional **round-trip** (kk→en) semantic match. Drop empties, length blowups, script errors (must be Cyrillic Kazakh).
- KazQAD already ships **~61k NQ triples machine-translated to Kazakh** — use directly.
- NLI is especially valuable: it teaches semantic distinctions cheaply. Use entailment pairs as positives, contradiction as hard negatives.

### 4.3 Cross-encoder distillation (§3.5)
Teacher = `BAAI/bge-reranker-v2-m3`. Distilling relevance judgments densifies weak/translated labels and is one of the most reliable gains.

### 4.4 LLM-generated synthetic query→passage pairs (Promptagator / Gecko / E5-mistral style)
Generate task-specific training data from the *target* Kazakh corpus (KazQAD's 800k Wikipedia passages, Kazakh news, gov, legal):
- **Two-stage (Gecko/E5-mistral):** (1) prompt a capable multilingual LLM to read a sampled Kazakh passage and emit a task description + a Kazakh query; (2) retrieve top-k with a first-pass retriever, then have the LLM **relabel the best positive and pick a good hard negative**.
- Prompt the LLM **in Kazakh** and require Cyrillic Kazakh output; generate diverse query types (short keyword, natural question, long descriptive).
- Filter with the same LaBSE/consistency checks. Synthetic data closes the domain gap that translated MS MARCO can't.

### 4.5 Cross-lingual alignment with kk↔ru / kk↔en parallel pairs
Add parallel bitext as positive pairs so Kazakh anchors into the shared multilingual space (and enables cross-lingual retrieval, which BGE-M3 is only mediocre at for Kazakh):
- **Sources:** OPUS (CCMatrix, CCAligned, Tatoeba), FLORES-200, WikiMatrix, government/news parallel corpora.
- Train these with MNRL/CMNRL as (kk, ru) and (kk, en) positive pairs, mixed in at ~10–20% of batches. Improves both monolingual Kazakh and cross-lingual scores; also a cheap regularizer against catastrophic forgetting of other languages.

### 4.6 Continued pretraining (MLM / RetroMAE) on Kazakh mono-corpus
Do this **before** contrastive FT **if the base is weak on Kazakh** (e.g. plain XLM-R). Modern bases (mmBERT, e5, bge-m3) already model Kazakh reasonably, so gains here are smaller — treat as optional:
- **MLM CPT:** HF `run_mlm.py` on a Kazakh corpus (OSCAR, CC-100 kk, Leipzig, Kazakh Wikipedia/news, ~1–2B tokens). Adapts token/subword statistics.
- **RetroMAE** (what BGE-M3 itself used) is *retrieval-oriented* CPT (masked auto-encoding with a weak decoder) and helps dense retrieval more than vanilla MLM. FlagEmbedding provides RetroMAE pretraining code.
- For low-resource, **bilingual CPT** (mix kk with related ru/tr batches, BALM/BJLM style) improves cross-lingual transfer.
- Keep it short (1–3 epochs); the contrastive stage is where retrieval quality is made.

> **Practical priority if time is limited:** 4.2 + 4.1 + 4.3 first (translated data → hard negs → distillation). Add 4.4/4.5 next. Do 4.6 only if the base underperforms on Kazakh MLM/probing.

---

## 5. RTX 3090 (24 GB) practicalities

### 5.1 What fits
All recommended bases (278 M–568 M params) **full-fine-tune** on a 24 GB 3090. Rough VRAM budget for a 560 M encoder:
- Params (bf16) ≈ 1.1 GB, grads (bf16) ≈ 1.1 GB, **AdamW states (fp32) ≈ 4.5 GB** (use **8-bit Adam** → ~1.1 GB), activations ≈ depends on batch/seq but **small with gradient checkpointing**.
- → ~7–9 GB fixed, leaving ~15 GB for activations/batch. Comfortable.

### 5.2 Full fine-tune vs LoRA/QLoRA for embedders
- **Full fine-tune is the default and usually best** for ≤600 M embedders — it fits easily, and embedders benefit from moving all params. Use it.
- **LoRA** is worth it when: you want to **preserve the multilingual base** (less catastrophic forgetting on other languages), run **longer sequences / bigger batches**, or later scale to a **7B LLM-based embedder** (then QLoRA is mandatory). LoRA embedders are viable (FlagEmbedding supports `--use_lora` for decoder embedders) but for a 300–570 M encoder the quality ceiling is slightly lower than full FT.
- **QLoRA** only needed for 7B+ embedders — not necessary here.

### 5.3 Precision & memory switches
- **bf16** (the 3090 supports it) — prefer over fp16 (no loss-scaling headaches). Set `bf16=True`.
- **`gradient_checkpointing=True`** — trade compute for memory; enables larger `mini_batch_size`/seq.
- **8-bit Adam** (`optim="adamw_bnb_8bit"`, needs `bitsandbytes`) — frees ~3 GB.
- **GradCache** via `Cached*` losses — the key to large effective batch (see §3.2).
- **`tf32`** matmul enabled for speed.

### 5.4 Realistic batch sizes & time
- Seq 512, full FT, bf16 + checkpointing: `mini_batch_size` 16–32; **effective batch 512–1024** via CMNRL. (Seq 256 → double these.)
- For `bge-m3`/e5-large expect **~1.5–3 it/s** at mini-batch 16, seq 512.
- **Time:** ~500k–1M pairs, 1–3 epochs ≈ **a few hours to ~1 day per run**. MLM/RetroMAE CPT on ~1B tokens ≈ 1–2 days (optional). Budget for 2–3 full iterations (mine → train → re-mine).

### 5.5 Tooling
- **Sentence Transformers v3+/v5** `SentenceTransformerTrainer` — best DX, native `Cached*`, `GISTEmbed`, `Matryoshka`, `mine_hard_negatives`, MTEB integration. **Primary choice.**
- **FlagEmbedding** (BGE finetune) — use when you want **BGE-M3's unified dense+sparse+colbert fine-tuning** and its **`m3_kd_loss` self-distillation** (`FlagEmbedding.finetune.embedder.encoder_only.m3`). Best if you fine-tune BGE-M3 itself and want to keep its multi-functionality.
- **PEFT** for LoRA, **accelerate** for launch config, **bitsandbytes** for 8-bit Adam, **faiss-gpu** for mining/eval.

---

## 6. End-to-end recipe (step by step)

### Step 0 — Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U "sentence-transformers>=5.0" "transformers>=4.44" datasets accelerate \
               peft bitsandbytes faiss-gpu mteb FlagEmbedding
# NLLB for translation:
pip install sacremoses
```

### Step 1 — Data prep (build the Kazakh pair pool)
1. **Native:** download `issai/kazqad-retrieval` and `issai/kazqad` (queries, 800k-passage corpus, qrels, +61k translated NQ triples).
2. **Translate EN sets** (MS MARCO / NQ / HotpotQA / AllNLI) to `kaz_Cyrl` with NLLB-200; filter by **LaBSE cosine > 0.7** + round-trip check.
3. **Parallel bitext:** pull kk↔ru, kk↔en from OPUS/FLORES/Tatoeba → positive pairs.
4. **Synthetic:** LLM-generate Kazakh queries from sampled corpus passages (two-stage Gecko/E5-mistral), filter.
5. Normalize everything to JSONL. For Sentence Transformers use columns `anchor,positive` (pairs) or `anchor,positive,negative_1..n`. For FlagEmbedding use `{"query","pos":[...],"neg":[...]}` (+`pos_scores`/`neg_scores` if distilling).

### Step 2 — (Optional) Continued pretraining on Kazakh
Only if base is weak on Kazakh. MLM:
```bash
python run_mlm.py --model_name_or_path jhu-clsp/mmBERT-base \
  --train_file kk_mono.txt --line_by_line --do_train \
  --per_device_train_batch_size 32 --gradient_checkpointing --bf16 \
  --max_seq_length 512 --num_train_epochs 2 --output_dir ckpt/kk-mlm
```
Prefer **RetroMAE** (FlagEmbedding) for a retrieval-oriented CPT if you have time.

### Step 3 — First-pass hard-negative mining
Use `mine_hard_negatives` (§3.3) with `intfloat/multilingual-e5-large` or `bge-m3` as the miner → produces `kk_triplets_minedHN.jsonl` (`n-tuple`). FlagEmbedding equivalent:
```bash
python -m FlagEmbedding.finetune.embedder.hn_mine \
  --input_file kk_train.jsonl --output_file kk_train_minedHN.jsonl \
  --range_for_sampling 10-100 --negative_number 10 \
  --use_gpu_for_searching --embedder_name_or_path BAAI/bge-m3
```

### Step 4 — (Big lever) Teacher scoring for distillation
Score every (query, pos/neg) with `BAAI/bge-reranker-v2-m3`; write `pos_scores`/`neg_scores` back into the JSONL (FlagEmbedding), or cache margins for ST `MarginMSELoss`.

### Step 5 — Contrastive fine-tune with Matryoshka (Sentence Transformers, the main run)
See config in §7. Loss = `CachedGISTEmbedLoss` (or `CachedMNRL`) wrapped in `MatryoshkaLoss`, bf16 + checkpointing + 8-bit Adam, effective batch 512–1024.

**Alternative — fine-tune BGE-M3 itself with unified + self-distillation (FlagEmbedding):**
```bash
torchrun --nproc_per_node 1 \
  -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
  --model_name_or_path BAAI/bge-m3 \
  --train_data ./kk_train_minedHN.jsonl \
  --knowledge_distillation True --kd_loss_type m3_kd_loss \
  --unified_finetuning True --use_self_distill True \
  --train_group_size 8 --query_max_len 128 --passage_max_len 512 \
  --temperature 0.02 --learning_rate 1e-5 --warmup_ratio 0.1 \
  --bf16 --gradient_checkpointing --per_device_train_batch_size 4 \
  --negatives_cross_device --normalize_embeddings True \
  --output_dir ckpt/kk-bge-m3-ft
```

### Step 6 — Evaluate head-to-head vs BGE-M3
Run the Kazakh MTEB task list (§1) on your checkpoint **and** `BAAI/bge-m3`. Also run a KazQAD-retrieval harness (encode 800k passages, retrieve, compute nDCG@10/MRR/Recall@100). Keep the checkpoint that wins retrieval without regressing classification/bitext.

### Step 7 — Iterate
Re-mine hard negatives **with your new model**, re-score with the teacher, retrain 1 epoch. One or two rounds usually adds a few points. Ship the winner; export Matryoshka dims (1024/512/256) for deployment flexibility.

---

## 7. Minimal training config / snippet (Sentence Transformers, 3090-tuned)

```python
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerModelCardData,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.losses import CachedGISTEmbedLoss, MatryoshkaLoss

torch.backends.cuda.matmul.allow_tf32 = True

# 1) Base model (Track A). For e5/bge, prompts help retrieval quality.
model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    model_card_data=SentenceTransformerModelCardData(language="kk", license="apache-2.0"),
)
model.gradient_checkpointing_enable()

# 2) Data: mined hard-negative triplets/n-tuples (cols: anchor, positive, negative_1..n)
ds = load_dataset("json", data_files="kk_triplets_minedHN.jsonl", split="train")
splits = ds.train_test_split(test_size=2000, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

# 3) Loss: guided false-negative filtering (guide=bge-m3) + GradCache + Matryoshka
guide = SentenceTransformer("BAAI/bge-m3")
inner = CachedGISTEmbedLoss(model, guide=guide, mini_batch_size=16, temperature=0.02)
loss  = MatryoshkaLoss(model, inner, matryoshka_dims=[1024, 768, 512, 256, 128, 64])

# 4) 3090-tuned training args (effective batch = per_device * grad_accum, negatives cached)
args = SentenceTransformerTrainingArguments(
    output_dir="ckpt/kaz-embed-v1",
    num_train_epochs=2,
    per_device_train_batch_size=256,      # GradCache holds only mini_batch_size at a time
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=2,        # effective in-batch negatives ~512
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True, fp16=False, tf32=True,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",               # 8-bit Adam via bitsandbytes
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps", eval_steps=200,
    save_strategy="steps", save_steps=200, save_total_limit=2,
    load_best_model_at_end=True,
    logging_steps=50,
    run_name="kaz-embed-v1",
)

trainer = SentenceTransformerTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    loss=loss,
)
trainer.train()
model.save_pretrained("ckpt/kaz-embed-v1/final")
```
Notes: if OOM, lower `mini_batch_size` (16→8) or `max_seq_length` (512→256), not the effective batch. For e5 bases, prepend `query: ` / `passage: `; for bge-m3 no prefix. Add cross-lingual (kk-ru/kk-en) pairs as a second dataset/loss in a dict to `train_dataset`/`loss` for multi-task training.

---

## 8. Biggest levers & pitfalls (to actually win)

**Levers (do these):**
1. **Hard negatives with false-negative filtering** (NV-Retriever margins + GISTEmbed guide). Highest ROI.
2. **Volume of clean translated triplets** (MS MARCO/NQ/NLI → kk, LaBSE-filtered). Second highest.
3. **Cross-encoder distillation** from `bge-reranker-v2-m3`. Reliable extra points.
4. **Large effective batch** via CachedMNRL/GradCache (more in-batch negatives).
5. **LLM synthetic queries** on the *target* corpus (closes domain gap).
6. **Cross-lingual anchoring** (kk-ru/kk-en) — improves Kazakh *and* cross-lingual, regularizes forgetting.
7. **Iterative mining** (re-mine with your own improving model).
8. **Matryoshka** for free flexible dims.

**Pitfalls (avoid these):**
- **False negatives** in translated/synthetic data silently cap quality → always filter (relative margin, `max_score`, GISTEmbed guide).
- **Contrastive FT alone can regress a strong base** → add distillation and/or keep a small in-domain multilingual mix.
- **Catastrophic forgetting** of other languages → mix in parallel/multilingual pairs, or use LoRA.
- **Evaluating only at the end** → wire up KazQAD + MMTEB-kk from day one and diff against BGE-M3 every run.
- **Prefix mismatch** (e5/gte need `query:`/`passage:`; bge-m3 doesn't) → keep encode-time prompts consistent between training and eval.
- **fp16 instability** on contrastive loss → use **bf16**.

---

## 9. Key references
- Sentence Transformers — Training/Loss overview: <https://sbert.net/docs/sentence_transformer/training_overview.html>, <https://sbert.net/docs/sentence_transformer/loss_overview.html>
- Training & Finetuning Embedding Models (HF blog): <https://huggingface.co/blog/train-sentence-transformers>
- `mine_hard_negatives` API: <https://www.sbert.net/docs/package_reference/util/hard_negatives.html>
- NV-Retriever (hard-negative mining best practices), arXiv 2407.15831: <https://arxiv.org/abs/2407.15831>
- Mitigating False Negatives in MNRL (HF blog): <https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training>
- GISTEmbed (World Bank): <https://github.com/worldbank/GISTEmbed>
- Matryoshka Embeddings (ST docs / HF blog): <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>
- BGE-M3 paper (arXiv 2402.03216) & model card: <https://arxiv.org/html/2402.03216v3>, <https://huggingface.co/BAAI/bge-m3>
- FlagEmbedding finetuning (embedder + BGE-M3 unified/self-distill): <https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder>
- Cross-encoder listwise distillation for dense retrieval, arXiv 2505.19274: <https://arxiv.org/html/2505.19274v1>
- Improving Text Embeddings with LLMs (E5-mistral), arXiv 2401.00368: <https://arxiv.org/html/2401.00368v3>
- Gecko (LLM-distilled embeddings), arXiv 2403.20327: <https://arxiv.org/pdf/2403.20327>
- mmBERT (modern multilingual encoder, low-resource curriculum), arXiv 2509.06888: <https://huggingface.co/jhu-clsp/mmBERT-base>
- MMTEB (multilingual benchmark), arXiv 2502.13595: <https://arxiv.org/abs/2502.13595>
- KazQAD (Kazakh ODQA/retrieval), LREC-COLING 2024: <https://huggingface.co/datasets/issai/kazqad-retrieval>, <https://github.com/IS2AI/KazQAD>
- NLLB-200 (kk translation): `facebook/nllb-200-distilled-1.3B`
