# Kazakh Embedding Model — Base Models & Baselines to Beat

**Goal:** build a Kazakh (kk) text embedding model that **beats BGE-M3** on Kazakh benchmarks, fine-tunable on a single **RTX 3090 (24 GB)**.

**Compiled:** 2026-09-20. All numbers taken from the cited primary sources. Anything not directly confirmed in a source is marked **(verify)**. Nothing here is fabricated.

---

## Part 1 — Baseline numbers to BEAT on Kazakh

There are three published Kazakh evaluations that matter. The headline bar is **BGE-M3 on KazQAD retrieval** (the hardest, least-saturated task).

### 1A. KazQAD retrieval — modern dense encoders (THE PRIMARY BAR)

Source: *"A Systematic Evaluation of LLMs and RAG for Kazakh Question Answering"*, MDPI Information 16(11):943 — [mdpi.com/2078-2489/16/11/943](https://www.mdpi.com/2078-2489/16/11/943) (repo: [github.com/Arailym-ray/KAZ-QA-RAG](https://github.com/Arailym-ray/KAZ-QA-RAG)). Evaluated zero-shot on **ISSAI/kazqad-retrieval**.

| Model | R@1 | R@10 | MRR | ΔCos (relevant vs non-relevant) |
|---|---|---|---|---|
| **BGE-M3** (the model to beat) | **0.641** | **0.929** | **0.746** | ~0.092 |
| Snowflake Arctic-Embed (v2.0) | 0.591 | 0.907 | ~0.69 (verify) | 0.099 (largest margin) |
| E5-base (multilingual-e5-base) | 0.562 | 0.859 | ~0.68 (verify) | ~0.085 |
| BM25 (baseline) | 0.489 | 0.807 | 0.591 | N/A |
| LaBSE | 0.365 | N/A (verify) | N/A (verify) | N/A |
| OpenAI text-embedding-3-large | 0.323 | N/A (verify) | N/A (verify) | N/A |

> **This is the number to beat: BGE-M3 R@1 = 0.641, R@10 = 0.929, MRR = 0.746 on KazQAD retrieval.**
> Note: text-embedding-3-large and LaBSE do *poorly* on Kazakh retrieval here (R@1 0.32–0.37) — Kazakh retrieval is not the same as generic multilingual STS.

**End-to-end RAG (context, not the retrieval bar):** best system = KazLLM-8B + Snowflake Arctic-Embed, answer correctness ≈ **0.76** (top-5); GPT-4o + text-embedding-3-large ≈ **0.65**. The paper explicitly notes "high retrieval metrics do not guarantee high QA accuracy" (Arctic-Embed wins downstream despite slightly lower recall than BGE-M3).

### 1B. KazQAD original dataset paper — classic/neural rerankers (older, weaker baselines)

Source: *KazQAD: Kazakh Open-Domain QA Dataset*, arXiv [2404.04487](https://arxiv.org/abs/2404.04487) (2024). Dataset: ~6,000 questions, ~12,000 relevance judgements, >800,000 Kazakh Wikipedia passages. These predate the modern dense encoders above — useful as a floor, not the real bar.

| IR system (Table 4) | NDCG@10 | MRR | R@100 |
|---|---|---|---|
| BM25 (title+text) | 0.1446 | 0.1443 | 0.4491 |
| BM25 (multi-field) | 0.1992 | 0.1961 | 0.5559 |
| BM25+Model1 (multi-field) | 0.2735 | 0.2723 | 0.6478 |
| + mBERT (zero-shot) | 0.3450 | 0.3344 | 0.6478 |
| + mBERT (fine-tuned) | 0.3672 | 0.3628 | 0.6478 |
| + XLM-R (zero-shot) | 0.3764 | 0.3654 | 0.6478 |
| + XLM-R (fine-tuned) | **0.3892** | **0.3822** | 0.6478 |

Best reported in the original paper: NDCG@10 = 0.389, MRR = 0.382 (BM25 + fine-tuned XLM-R reranker).

### 1C. KazakhTextDuplicates v2.0 (dedup / STS / retrieval)

Source: *KazakhTextDuplicates: A Controlled Multi-Regime Benchmark…*, MDPI Data 11(6):133 — [mdpi.com/2306-5729/11/6/133](https://www.mdpi.com/2306-5729/11/6/133). Dataset: [Arailym-tleubayeva/KazakhTextDuplicates](https://huggingface.co/datasets/Arailym-tleubayeva/KazakhTextDuplicates) (25,922 kk text pairs; 7 deterministic regimes: exact, contextual reformulation, paraphrase, partial overlap, 3 noise levels). All models evaluated **zero-shot**.

**Retrieval ranking (Table 11) — SATURATED, weak discriminator:**

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| **BGE-M3** | **0.651** | 0.964 | 0.998 | **0.806** |
| LaBSE | 0.648 | 0.963 | 0.998 | 0.805 |
| Multilingual-E5-Large | 0.647 | 0.963 | 0.998 | 0.805 |
| OpenAI text-embedding-3-large | 0.648 | 0.963 | 0.998 | 0.804 |
| KazEmbed-v5 | 0.647 | 0.963 | 0.998 | 0.803 |

(All within 0.003 MRR — this task does not separate models. Do not optimize for it.)

**STS regression (Table 7) — the meaningful similarity bar:**

| Model | Pearson | Spearman | MSE |
|---|---|---|---|
| OpenAI text-embedding-3-large | 0.510 | **0.652** | 0.075 |
| KazEmbed-v5 | 0.401 | 0.524 | 0.108 |
| **BGE-M3** | 0.396 | **0.538** | 0.109 |
| Multilingual-E5-Large | 0.356 | 0.480 | 0.120 |
| LaBSE | 0.355 | 0.479 | 0.120 |

> **STS bar to beat: BGE-M3 Spearman = 0.538** (best open-source). Overall ceiling is OpenAI at 0.652 — a large open gap to close.

**Zero-shot duplicate-regime classification (Table 8) — hard for everyone (random ≈ 0.143):**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| OpenAI text-embedding-3-large | 0.155 | 0.066 |
| BGE-M3 | 0.152 | 0.058 |
| KazEmbed-v5 | 0.152 | 0.058 |
| LaBSE | 0.145 | 0.048 |
| Multilingual-E5-Large | 0.141 | 0.037 |

(Barely above chance for all — this is a stress test, not a primary target.)

### 1D. Comparative analysis QA (informatics_kaz)

Source: *"Comparative Analysis of Embedding Models for Matching Questions and Contexts in the Kazakh Language"* — ResearchGate [publication/396213005](https://www.researchgate.net/publication/396213005). Dataset: [Kundyzka/informatics_kaz](https://huggingface.co/datasets/Kundyzka/informatics_kaz) (~7,700 kk question–context pairs, CS/informatics). Metrics: Accuracy@1, MRR, ROC-AUC. Models: TF-IDF, mMiniLM, LaBSE, GTE-multilingual-base, multilingual-e5-small, Arctic-Embed v2.0.

| Model | Accuracy@1 | MRR | ROC-AUC |
|---|---|---|---|
| Best embedders (GTE-multilingual-base & Arctic-Embed v2.0 reported as top) | **0.59–0.63** (range) | (verify) | (verify) |
| TF-IDF baseline | ~0.50 | (verify) | (verify) |

> **Full page was CAPTCHA-blocked** — only the abstract's ranges are confirmed. Exact per-model Accuracy@1/MRR/ROC-AUC = **(verify)** by pulling the PDF directly. Confirmed takeaways: embedders reach Acc@1 0.59–0.63 vs TF-IDF 0.50; **Alibaba GTE-multilingual-base and Snowflake Arctic-Embed v2.0 won**. Note BGE-M3 was *not* in this study.

### Summary of hard targets

| Benchmark | Metric | BGE-M3 value to beat |
|---|---|---|
| **KazQAD retrieval** (primary) | R@1 / R@10 / MRR | **0.641 / 0.929 / 0.746** |
| KazakhTextDuplicates STS | Spearman | **0.538** (open ceiling 0.652) |
| KazakhTextDuplicates retrieval | MRR | 0.806 (saturated — ignore) |

---

## Part 2 — Candidate base models to fine-tune (single RTX 3090, 24 GB)

Legend: **dim** = dense embedding dimension; **ctx** = max sequence length; **MRL** = Matryoshka (truncatable dims).

| # | HF repo ID | Params | dim | ctx | Backbone | Kazakh coverage | License | 3090 fit |
|---|---|---|---|---|---|---|---|---|
| 1 | `BAAI/bge-m3` | 568M | 1024 | 8192 | XLM-R-large + RetroMAE | Yes (via XLM-R 100+ langs; 194-lang pretrain) | **MIT** | Full FT feasible w/ grad-ckpt |
| 2 | `Alibaba-NLP/gte-multilingual-base` | 305M | 768 (MRL) | 8192 | Custom encoder + RoPE | **Yes — 1,020M kk tokens in pretrain** (75 langs) | **Apache-2.0** | Full FT easy, large batch |
| 3 | `intfloat/multilingual-e5-large` | 560M | 1024 | 512 | XLM-R-large | Yes (XLM-R 100 langs; kk not called out) | **MIT** | Full FT feasible |
| 4 | `intfloat/multilingual-e5-base` | 278M | 768 | 512 | XLM-R-base | Yes (XLM-R) | **MIT** | Full FT easy |
| 5 | `Snowflake/snowflake-arctic-embed-l-v2.0` | 568M | 1024 (MRL→256) | 8192 (RoPE) | **BAAI/bge-m3-retromae** (XLM-R-large) | Yes, 74 langs (kk in XLM-R; verify in-mix) | **Apache-2.0** | Full FT feasible w/ grad-ckpt |
| 6 | `jinaai/jina-embeddings-v3` | 570M | 1024 (MRL→32) | 8192 | XLM-R + task-LoRA | 30+ langs; **kk not confirmed** (verify) | **CC-BY-NC-4.0 (non-commercial!)** | FT feasible; LoRA-based |
| 7 | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | up to 1024 (MRL) | 32K | Qwen3 decoder | 100+ langs, instruction-aware (kk verify) | **Apache-2.0** | Full FT (small batch) or LoRA |
| 8 | `Qwen/Qwen3-Embedding-4B` | 4B | up to 2560 (MRL) | 32K | Qwen3 decoder | 100+ langs (kk verify) | **Apache-2.0** | **QLoRA (4-bit) only**; small batch |
| 9 | `sentence-transformers/LaBSE` | 471M (~0.5B) | 768 | 256/512 | BERT dual-encoder | **Yes — 109 langs incl. Kazakh** | **Apache-2.0** | Full FT easy |
| 10 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 278M | 768 | **128** | XLM-R-base (distilled) | **No — 50-lang list excludes kk** (verify) | Apache-2.0 | Easy, but weak base |
| 11 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | 384 | 128 | MiniLM (distilled) | Same 50 langs, **kk excluded** (verify) | Apache-2.0 | Trivial |
| 12 | `kz-transformers/kaz-roberta-conversational` | 83.5M | 768 | 512 | RoBERTa (6 layers), vocab 52k | **Native Kazakh** (25 GB kk corpus) MLM only | Apache-2.0 | Trivial — but needs ST training from scratch |

### Per-model notes (why good / bad base)

- **`BAAI/bge-m3` (good, safe):** the exact model we must beat. Continue-fine-tuning it on Kazakh in-domain data with hard negatives is the *surest* way to not regress and to clear the bar — but it is a somewhat circular win. MIT, 8192 ctx, dense+sparse+ColBERT heads.
- **`Alibaba-NLP/gte-multilingual-base` (BEST small base):** the only compact model with **documented, substantial Kazakh pretraining (1.02B kk tokens)**. Already won the informatics_kaz comparative study. 305M means you can **full-fine-tune with large contrastive batches on a 3090** (large batch = more/harder in-batch negatives = the single biggest lever for beating BGE-M3). Apache-2.0, MRL, 8192 ctx. Strong specialization headroom.
- **`intfloat/multilingual-e5-large` (good):** solid XLM-R-large retriever, MIT, but **ctx only 512** and requires `query:`/`passage:` prefixes. Kazakh inherited from XLM-R (not explicitly optimized).
- **`intfloat/multilingual-e5-base` (decent, light):** the exact "E5-base" that scored R@1 0.562 on KazQAD. Cheap to train; lower ceiling than large.
- **`Snowflake/snowflake-arctic-embed-l-v2.0` (strong ceiling):** #2 on KazQAD and best downstream RAG. Built on **the same bge-m3-retromae backbone**, Apache-2.0, MRL→256, 8192 ctx. Fine-tuning this on Kazakh targets the highest absolute retrieval quality.
- **`jinaai/jina-embeddings-v3` (avoid for shipping):** **CC-BY-NC-4.0 — non-commercial only.** Kazakh not confirmed. Skip unless the project is research-only.
- **`Qwen/Qwen3-Embedding-0.6B` (experimental):** decoder-based, instruction-aware, MRL, 32K ctx, Apache-2.0, tops MTEB multilingual (8B). 0.6B is 3090-trainable (small batch / LoRA). Decoder embedders can be strong but contrastive training wants big batches, which is harder here.
- **`Qwen/Qwen3-Embedding-4B` (hard on one 3090):** QLoRA-only; small batches hurt contrastive learning. High ceiling but risky/slow on a single 3090.
- **`sentence-transformers/LaBSE` (weak for retrieval):** explicitly covers Kazakh (109 langs), Apache-2.0 — but scored only R@1 0.365 on KazQAD; it is a bitext/STS model, not a strong asymmetric retriever. Weak base for beating BGE-M3 on retrieval.
- **`paraphrase-multilingual-*` (bad base):** 50-language list **excludes Kazakh**, ctx only 128. Poor fit — do not use.
- **`kz-transformers/kaz-roberta-conversational` (from-scratch ST base):** the only *natively Kazakh* encoder here (25 GB kk corpus), but tiny (83.5M, 6 layers, MLM-only) and never trained as a sentence encoder. Would need full contrastive ST training from scratch — high risk, low ceiling vs. a multilingual base that already embeds well. Best used as a research baseline / distillation student, not the primary bet.
- **KazByte** — arXiv [2603.27859](https://arxiv.org/abs/2603.27859) "Adapting Qwen models to Kazakh via Byte-level Adapter": adapts **Qwen2.5-7B** to Kazakh with a frozen byte-level adapter + attention-layer tuning. It is a **research proposal for an LLM (no empirical results, not an embedding model)**. Relevant only as a *Kazakh-adaptation technique*, not a usable embedding base.

---

## Part 3 — Training recipes of the top open models (what we compete against)

### BGE-M3 (arXiv [2402.03216](https://arxiv.org/abs/2402.03216))
- **Base:** XLM-RoBERTa-large, further pretrained with **RetroMAE**; ctx extended 512 → 8192; dim 1024.
- **Data:** ~**1.2B** unsupervised text pairs across **194 languages**; then **~1.6M+** labeled pairs for fine-tuning (EN 1.1M, ZH 386.6K, multilingual 88.9K, long-doc 41.4K); +41.4K GPT-3.5-synthesized long-doc pairs.
- **Objective:** InfoNCE contrastive loss; **ANCE-style hard-negative mining, 7 negatives/query**, plus large in-batch negatives (distributed across GPUs).
- **Signature trick — self-knowledge distillation:** three retrieval heads (dense [CLS], sparse/lexical via ReLU term weights, multi-vector ColBERT-style) teach each other — the sum of their relevance scores is the teacher signal for each head's loss. Enables one model to do dense + sparse + multi-vector.
- **Batch:** up to 1,152 at ≤500 tokens; split-batch + grad-checkpointing to reach effective ~3,840 at 8192 tokens.

### multilingual-e5-large (tech report arXiv [2402.05672](https://arxiv.org/abs/2402.05672))
- **Base:** XLM-RoBERTa-large (24 layers, dim 1024, ctx 512), MIT.
- **Stage 1 — weakly-supervised contrastive pretraining:** ~**5B** text pairs (filtered mC4 ~1B, NLLB translations ~2.4B, CC-News 400M, Reddit 800M, Wikipedia 150M, S2ORC, StackExchange, xP3, SBERT data). InfoNCE with in-batch negatives, large batch.
- **Stage 2 — supervised fine-tuning:** ~**2.1M** labeled pairs (MS MARCO, NQ, TriviaQA, MIRACL, Mr. TyDi, NLI, etc.) with mined hard negatives + knowledge distillation from a cross-encoder reranker.
- **Requires prefixes:** `query:` / `passage:`.

### mGTE / gte-multilingual-base (arXiv [2407.19669](https://arxiv.org/abs/2407.19669))
- **Base encoder:** trained from scratch, 12 layers, hidden 768, **RoPE** positions, unpadding, 305M params, ctx 8192.
- **Stage 1 — MLM:** MLM-2048 (250k steps, RoPE base 10k) → MLM-8192 (30k steps, RoPE base 160k) on **1,028B tokens across 75 languages** — **Kazakh included (1,020M kk tokens)**.
- **Stage 2 — weakly-supervised contrastive:** **~2.94B** pairs, batch **16,384**, 240k steps.
- **Stage 3 — supervised contrastive:** ~4.52M high-quality pairs (MS MARCO, MIRACL, MLDR…), 10 epochs, with hard negatives; MRL for elastic dims (32…768).

**Competitive takeaway:** all three win via (a) huge weakly-supervised contrastive pretraining and (b) supervised fine-tuning with **mined hard negatives + large in-batch negatives** (+ distillation). We cannot out-scale them globally — but they spend almost no capacity on Kazakh specifically. **Our edge = Kazakh-specialized supervised fine-tuning with well-mined Kazakh hard negatives**, on a base that already embeds Kazakh well.

---

## Part 4 — Recommendation

**To beat zero-shot BGE-M3 on Kazakh by fine-tuning on one RTX 3090, pick a base that (a) already handles Kazakh, (b) is small enough for large-batch contrastive full fine-tuning, and (c) is permissively licensed. Two picks:**

### Primary: `Alibaba-NLP/gte-multilingual-base` (305M, Apache-2.0)
- **Only compact model with documented heavy Kazakh pretraining (1.02B kk tokens).** Already the top embedder in the informatics_kaz comparative study.
- At 305M it **full-fine-tunes on a 3090 with large batches** (bf16 + grad-checkpointing), which maximizes in-batch/hard negatives — the biggest lever for beating BGE-M3.
- 8192 ctx, MRL, Apache-2.0 → deployable and flexible.
- Plan: supervised contrastive FT on Kazakh (KazQAD-retrieval train, informatics_kaz, translated MS-MARCO/MIRACL-kk, KazakhTextDuplicates STS) with **mined hard negatives**; distill from BGE-M3/Arctic teacher scores if budget allows.

### Secondary / highest ceiling: `Snowflake/snowflake-arctic-embed-l-v2.0` (568M, Apache-2.0)
- Already **#2 on KazQAD and best downstream RAG**, same bge-m3-retromae backbone, MRL→256, 8192 ctx.
- Fine-tuning the strongest existing multilingual retriever on Kazakh gives the **highest absolute chance of clearing R@1 0.641 / MRR 0.746**. On a 3090 use grad-checkpointing (batch ~16–32 at 512 tokens) or LoRA.

### Safe fallback: continue-fine-tune `BAAI/bge-m3` on Kazakh
- Starting from the model we must beat almost guarantees a win over zero-shot BGE-M3 (MIT, 8192 ctx). Use it as a floor/sanity check and as a hard-negative + distillation teacher for the gte/Arctic students.

### Skip / deprioritize
- `jina-embeddings-v3` — **non-commercial license**, Kazakh unconfirmed.
- `paraphrase-multilingual-*` — **Kazakh not in the 50-lang list**, ctx 128.
- `Qwen3-Embedding-4B` — QLoRA-only on a 3090; small batches undercut contrastive training. `Qwen3-Embedding-0.6B` is a fair *experimental* alternative if a decoder embedder is desired.
- `kaz-roberta-conversational` — native Kazakh but tiny and not an ST model; research baseline only.

### Numbers our fine-tuned model must clear
- **KazQAD retrieval (primary): R@1 > 0.641, R@10 > 0.929, MRR > 0.746.**
- **KazakhTextDuplicates STS: Spearman > 0.538** (stretch goal: approach OpenAI's 0.652).
- KazakhTextDuplicates retrieval MRR (~0.806) is saturated — not a useful target.

---

## Sources
- KazakhTextDuplicates v2.0 — MDPI Data 11(6):133: https://www.mdpi.com/2306-5729/11/6/133 ; dataset https://huggingface.co/datasets/Arailym-tleubayeva/KazakhTextDuplicates
- KazQAD RAG eval — MDPI Information 16(11):943: https://www.mdpi.com/2078-2489/16/11/943 ; code https://github.com/Arailym-ray/KAZ-QA-RAG
- KazQAD dataset paper — arXiv 2404.04487: https://arxiv.org/abs/2404.04487
- Comparative analysis (informatics_kaz) — ResearchGate 396213005 (CAPTCHA-blocked; abstract only): https://www.researchgate.net/publication/396213005
- BGE-M3 — arXiv 2402.03216: https://arxiv.org/abs/2402.03216
- multilingual-e5 — arXiv 2402.05672 ; https://huggingface.co/intfloat/multilingual-e5-large
- mGTE / gte-multilingual — arXiv 2407.19669 ; https://huggingface.co/Alibaba-NLP/gte-multilingual-base
- Snowflake Arctic-Embed v2.0 — arXiv 2412.04506 ; https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0
- jina-embeddings-v3 — arXiv 2409.10173 ; https://huggingface.co/jinaai/jina-embeddings-v3
- Qwen3-Embedding — https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- LaBSE — https://huggingface.co/sentence-transformers/LaBSE
- kaz-roberta-conversational — https://huggingface.co/kz-transformers/kaz-roberta-conversational
- KazByte — arXiv 2603.27859: https://arxiv.org/abs/2603.27859
