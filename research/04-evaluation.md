# 04 — Evaluation: How We Measure "Beating BGE-M3" for Kazakh Embeddings

Owner research note. Date: 2026-09-20. Goal: define exactly HOW we measure success ("we beat BGE-M3 on Kazakh") and design the eval harness. Items marked **[verify]** need a second confirmation against the primary PDF/table (some primary sources block automated fetch, e.g. MDPI 403).

Environment target for running evals: single RTX 3090 (24 GB). All baseline embedders below fit comfortably.

---

## TL;DR — The bar to beat

BGE-M3 (`BAAI/bge-m3`) is the reference. Headline benchmark = **KazakhTextDuplicates v2.0**. Targets to exceed:

| Benchmark | Metric | BGE-M3 number to beat | Source |
|---|---|---|---|
| KazakhTextDuplicates v2.0 — Retrieval | **MRR** (HEADLINE) | **0.806** | MDPI 2306-5729/11/6/133 |
| KazakhTextDuplicates v2.0 — Retrieval | Recall@1 | 0.651 | same |
| KazakhTextDuplicates v2.0 — STS | Spearman | 0.538 **[verify]** | same (see caveat below) |
| KazQAD — Retrieval | nDCG@10 | measure ourselves (paper baseline = 0.389 for its own retriever) | arXiv 2404.04487 |

Caveat on STS Spearman: the paper reports OpenAI `text-embedding-3-large` as the STS leader (Pearson 0.510, Spearman **0.652**, MSE 0.075). BGE-M3's STS Spearman = **0.538** is the caller-provided figure and is consistent with BGE-M3 leading retrieval but not STS. Confirm the exact BGE-M3 STS row from the MDPI table. **[verify]**

Note on margins: on v2.0 retrieval the top models are almost tied — BGE-M3 MRR 0.806, LaBSE and multilingual-e5-large 0.805, KazEmbed-v5 0.803. A credible "beat" needs a clear gap (target MRR >= ~0.83) and/or statistical significance across seeds, because a 0.001 delta is noise.

---

## 1) The benchmarks we must win on

### 1.1 KazakhTextDuplicates (MDPI Data, 2306-5729/11/6/133) — HEADLINE

- Paper: "KazakhTextDuplicates: A Controlled Multi-Regime Benchmark for Semantic Deduplication, Semantic Similarity, and Retrieval in Kazakh", MDPI *Data* 11(6):133. URL: https://www.mdpi.com/2306-5729/11/6/133 (open access; blocks automated fetch — read in browser).
- HF dataset: **`Arailym-tleubayeva/KazakhTextDuplicates`** — https://huggingface.co/datasets/Arailym-tleubayeva/KazakhTextDuplicates
  - What is currently visible on the HF page: single config `default`, single split `train`, **25,922 rows**, 217 MB, license **CC-BY-4.0**. Columns: `id`, `content`, `category`, `language` (`kk`), `type_duplicate`, `modified_content`. This is the **v1.0** natural/diagnostic dataset.
  - The paper describes a **two-version** design: **v1.0** = diagnostic natural dataset; **v2.0** = controlled benchmark. v2.0 is released in **two formats**: a **pair-level format** (paired texts + discrete regime labels; for duplicate classification + retrieval) and an **STS format** (continuous similarity scores; for regression STS). **[verify]** whether v2.0 lives in an additional config/split of the same repo or a separate revision — inspect all revisions/branches of the repo; if absent, request from authors.

- **Seven semantic duplication regimes** (deterministic transformations, each pair has a regime label + predefined similarity score):
  1. Exact duplication
  2. Contextual reformulation
  3. Paraphrasing
  4. Partial semantic overlap
  5. Character-level noise — level 1 (low)
  6. Character-level noise — level 2 (medium)
  7. Character-level noise — level 3 (high)

- **Three evaluation tasks + metrics**:
  - **Duplicate classification** (pair-level, regime labels): F1 / classification metrics **[verify exact metric]**.
  - **Retrieval**: **MRR, Recall@1, Recall@5, Recall@10**. (Paper note: on v2.0 all models drop sharply on Recall@1/MRR vs v1.0 while Recall@5/@10 stay high — the gold candidate is usually top-few but not always rank 1. So Recall@1/MRR is where differentiation happens.)
  - **STS** (regression on continuous scores): **Spearman, Pearson, MSE**.

- **BGE-M3 bar (v2.0)**: MRR **0.806** (highest), Recall@1 **0.651** (highest), STS Spearman **0.538 [verify]**.
- Nearby competitors (v2.0 retrieval MRR): LaBSE **0.805**, multilingual-e5-large **0.805**, KazEmbed-v5 **0.803**. STS leader: OpenAI text-embedding-3-large (Spearman 0.652).

### 1.2 KazQAD (arXiv 2404.04487) — secondary retrieval benchmark

- Paper: "KazQAD: Kazakh Open-Domain Question Answering Dataset", arXiv 2404.04487 (ISSAI, Nazarbayev Univ). PDF: https://arxiv.org/pdf/2404.04487 | HF paper page: https://huggingface.co/papers/2404.04487
- Code/data: GitHub **https://github.com/IS2AI/KazQAD** (has a `baselines/` dir). License **CC BY-SA 4.0**.
- HF datasets:
  - **`issai/kazqad`** — full dataset (reading comprehension + full ODQA).
  - **`issai/kazqad-retrieval`** — retrieval/IR format (BEIR-style): subsets `corpus` (Kazakh Wikipedia passages) + `queries-and-passages` (relevance judgements). Splits **train / validation / test**. License cc-by-sa-4.0.
- Sizes: ~**5,964 questions** (train 3,487 / val 548 / **test 1,929**); corpus **>800,000** Kazakh Wikipedia passages; **~11,821 qrels** (train 3,893 pos + 3,558 neg; val 769 pos + 229 neg; **test 2,718 pos + 653 neg**); plus ~61,000 machine-translated NQ triples (supplementary/training only).
- **Retrieval task + metrics**: **nDCG@10** and **MRR** (recall reported too). Paper's own baseline retriever: **nDCG@10 = 0.389, MRR = 0.382** (this is the paper's retriever, NOT BGE-M3 — do not treat 0.389 as the bar; measure BGE-M3 ourselves on the same corpus/qrels). Other settings for context: reading comprehension EM 38.5 / F1 54.2; full ODQA EM 17.8 / F1 28.7.
- Not in the `mteb` library → implement as a custom `AbsTaskRetrieval` (main_score `ndcg_at_10`) reading `issai/kazqad-retrieval`, OR reuse the repo's BM25/DPR baseline harness.

### 1.3 "Comparative analysis of embedding models for Kazakh" QA benchmark

- Paper: "Comparative Analysis of Embedding Models for Matching Questions and Contexts in the Kazakh Language" (ResearchGate pub. 396213005, 2025). https://www.researchgate.net/publication/396213005
- Dataset: **`Kundyzka/informatics_kaz`** — ~**7,700 question–context pairs**, computer-science/informatics domain. (Model card sibling: `Kundyzka/bert-base-multilingual-informatics-kaz`.)
- Models compared: **TF-IDF** (baseline), **mMiniLM**, **LaBSE**, **Alibaba GTE-multilingual-base**, **`intfloat/multilingual-e5-small`**, **Snowflake Arctic-Embed v2.0**.
- **Metrics: Accuracy@1, MRR, ROC-AUC.**
- Results: embedding models **Accuracy@1 ~0.59–0.63** vs TF-IDF 0.50; best = **GTE-multilingual-base** and **Snowflake Arctic-Embed v2.0** (zero-shot, no Kazakh fine-tuning). Per-model MRR / ROC-AUC values: **[verify from PDF]**.
- Related sibling work (use for context, not as our bar): MDPI *Information* 16(11):943 "A Systematic Evaluation of LLMs and RAG for Kazakh QA" (https://www.mdpi.com/2078-2489/16/11/943), repo `Arailym-ray/KAZ-QA-RAG` (https://github.com/Arailym-ray/KAZ-QA-RAG).
- Use: good third data point (a distinct QA-matching setup with ROC-AUC). We should run BGE-M3 + our model here too. Note it is small/domain-narrow (CS), so treat as supporting, not headline.

### 1.4 MTEB / MMTEB — does it have Kazakh?

- MMTEB paper: arXiv **2502.13595** ("Massive Multilingual Text Embedding Benchmark"). Library: `mteb` (checked **v2.21.0**, https://github.com/embeddings-benchmark/mteb).
- **There is NO Kazakh-specific text task in `mteb`.** Kazakh appears only as a **language subset** inside multilingual tasks. Text tasks with a Kazakh subset (exact task names + subset code):

  | MTEB task name | Type | Kazakh subset | Main metric |
  |---|---|---|---|
  | `BelebeleRetrieval` | Retrieval | `kaz_Cyrl` | ndcg_at_10 |
  | `WebFAQRetrieval` | Retrieval | `kaz` / `kaz-Cyrl` | ndcg_at_10 |
  | `SIB200Classification` | Classification | `kaz_Cyrl` | accuracy |
  | `TurkicClassification` | Classification | `kaz-Cyrl` | accuracy |
  | `CyrillicTurkicLangClassification` | Classification | `kaz-Cyrl` | accuracy |
  | `SIB200ClusteringS2S` | Clustering | `kaz_Cyrl` | v_measure |
  | `FloresBitextMining` | BitextMining | `kaz_Cyrl` | f1/accuracy |
  | `NTREXBitextMining` | BitextMining | `kaz_Cyrl` | f1/accuracy |
  | `Tatoeba` | BitextMining | `kaz-Cyrl` / `kaz-eng` | f1/accuracy |
  | `WebFAQBitextMiningQuestions` | BitextMining | `kaz-Cyrl` / `kaz-tur` | f1/accuracy |

  (Excluded audio/image tasks that also list Kazakh: `SIBFLEURS` (audio), `MVLSIBSent2Img` (image reranking), `CommonVoiceMini17A2TRetrieval`, `FleursA2TRetrieval` — not relevant to a text embedder.)

- Most decision-relevant MTEB tasks for us: **`BelebeleRetrieval` (kaz_Cyrl)** and **`WebFAQRetrieval` (kaz)** for retrieval; `SIB200Classification`/`SIB200ClusteringS2S` for classification/clustering breadth. Use these as **standardized, leakage-controlled cross-checks** alongside the Kazakh-native benchmarks.
- **KazQAD and KazakhTextDuplicates are not registered in `mteb`** → we add them as custom `mteb` tasks (see §2) so all evals run through one harness.

### 1.5 TUMLU (Turkic) — relevance

- Paper: arXiv **2502.11020**, ACL 2025. Native Turkic MMLU: 8 languages incl. **Kazakh**, **38,139** multiple-choice questions, 11 school subjects, Latin/Cyrillic/Arabic scripts. `TUMLU-mini` = balanced, manually verified subset + eval scripts released. HF: https://huggingface.co/papers/2502.11020
- **Relevance to us: LOW / out of scope for the headline claim.** TUMLU is a **generative-LLM knowledge/reasoning MC benchmark**, not a retrieval/STS/embedding benchmark. It does not measure embedding quality directly. (Same for the sibling **KazMMLU**, arXiv 2502.12829.)
- Optional only: it could be repurposed as a zero-shot "match the correct answer choice by embedding similarity" retrieval proxy, but that is a weak, non-standard signal. Do not include in the "we beat BGE-M3" claim. Mention as a downstream-utility footnote if we later build a Kazakh RAG demo.

---

## 2) Eval tooling

### 2.1 Install / environment

```bash
pip install "mteb==2.21.0" sentence-transformers torch --upgrade
# bge-m3 dense via sentence-transformers or FlagEmbedding; LaBSE + e5 via sentence-transformers.
```
3090 (24 GB) is enough for `bge-m3` (568M), `multilingual-e5-large` (560M), `LaBSE` (471M); use fp16 + batch 32–64.

### 2.2 Run `mteb` against a local sentence-transformers model

Three steps: define model → select tasks → evaluate (`mteb` v2 API).

```python
import mteb

# 1) MODEL — always prefer mteb.get_model so model-specific PROMPTS get applied
#    (critical for e5: it auto-adds "query: " / "passage: "; and for bge instruction).
model = mteb.get_model("intfloat/multilingual-e5-large")   # baseline
# our own local model:
# model = mteb.get_model("/home/oysyn/kaz-embed/checkpoints/kaz-embed-v1")
#   -> falls back to SentenceTransformer(path) if not registered in mteb.

# 2) TASKS — pick the Kazakh subset of multilingual tasks
tasks = mteb.get_tasks(
    tasks=["BelebeleRetrieval", "WebFAQRetrieval", "SIB200Classification", "SIB200ClusteringS2S"],
    languages=["kaz"],   # ISO 639-3; restricts multilingual tasks to Kazakh subset
)

# 3) EVALUATE
results = mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 32})
```

CLI equivalent:
```bash
mteb run -m intfloat/multilingual-e5-large -t BelebeleRetrieval --output-folder results/
```

Notes from the mteb docs (v2.21.0):
- `mteb.get_model(name)` returns the mteb reference implementation if one exists, else `SentenceTransformer(name)` / `CrossEncoder(name)`. Using the raw `SentenceTransformer` for e5 **skips the required prompts** and understates it — always go through `get_model` for baselines.
- `encode_kwargs` passes through to `.encode()` (e.g. `{"batch_size": 32}`; add `{"normalize_embeddings": True}` where relevant, though mteb applies each model's declared similarity function).
- Filtering: `mteb.get_tasks(task_types=["Retrieval"], languages=["kaz"])`, `mteb.get_task("BelebeleRetrieval", languages=["kaz"])`.

### 2.3 Custom model wrapper (EncoderProtocol)

If our model needs bespoke prompt logic, implement `encode` per `mteb.models.EncoderProtocol`:
```python
class KazEmbed:
    def encode(self, inputs, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs):
        # prompt_type is PromptType.query or PromptType.passage for retrieval
        # apply our own prefix convention here, then L2-normalize, return np.ndarray
        ...
```
Simplest path: ship the model as a standard `sentence-transformers` dir (with `prompts` in config) and register a `ModelMeta` so `get_model` applies our prompts consistently.

### 2.4 Reproduce KazakhTextDuplicates eval

Implement three custom mteb tasks over `Arailym-tleubayeva/KazakhTextDuplicates` (v2.0 formats):
- `AbsTaskRetrieval` (main_score `mrr` and report `recall_at_1/5/10`) — pair-level format: treat one side as query, the matching side as gold passage, add distractors.
- `AbsTaskSTS` (main_score `spearman`; also `pearson`, and MSE) — STS format (continuous scores).
- `AbsTaskPairClassification` — pair-level regime labels (duplicate vs not / per-regime).

Custom task skeleton (from mteb `docs/contributing/adding_a_dataset.md`):
```python
from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

class KazakhTextDuplicatesSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KazakhTextDuplicatesSTS",
        type="STS",
        main_score="spearman",
        dataset={"path": "Arailym-tleubayeva/KazakhTextDuplicates", "revision": "<pin>"},
        eval_splits=["test"], eval_langs=["kaz-Cyrl"],
        # ... min/max scores, columns ...
    )
```
**Reproduce the authors' exact numbers first** (protocol = their candidate pool, similarity fn, score scaling). Only once our BGE-M3 re-run matches their 0.806/0.651 within tolerance do we trust our harness. Pin dataset `revision`.

### 2.5 Reproduce KazQAD eval

Option A (preferred, unified harness): custom `AbsTaskRetrieval` (main_score `ndcg_at_10`) reading `issai/kazqad-retrieval` (`corpus` + `queries-and-passages` + qrels), eval split `test` (1,929 queries, 800k corpus). Run BGE-M3, e5, LaBSE, ours through it.
Option B (sanity check): use `IS2AI/KazQAD` `baselines/` (BM25 + dense) to confirm we reproduce paper's nDCG@10=0.389 for the paper's retriever before trusting our numbers.
Corpus is 800k passages — index once, reuse embeddings (mteb caches; or precompute with FAISS on the 3090).

### 2.6 Building a clean HELD-OUT Kazakh eval set (anti-leakage)

1. **Freeze external test splits as pure eval**: KazQAD `test`, KazakhTextDuplicates v2.0, MTEB Kazakh subsets. **Never** include these (or their source docs) in training or model selection.
2. **Dedup training data against every eval set**: exact-match + MinHash/LSH near-dup filtering (e.g. `datasketch`) of the training corpus against all eval texts (and their source Wikipedia passages / UNT items / the KazakhTextDuplicates source corpus). Log removed counts.
3. **Watch the translated-NQ overlap**: KazQAD ships ~61k MT-translated NQ triples intended for *training* — fine to train on, but confirm none leak into the KazQAD `test` gold passages.
4. **Build a fresh in-house held-out set** from **post-training-cutoff** Kazakh sources (news, gov, forums) to detect memorization: ~500–1000 query/passage pairs + ~300 STS pairs, annotated in-house; keep offline, use only for final report.
5. **Split hygiene**: any Kazakh corpus we scrape for training is dedup'd at the document level so no train/dev/test triplet shares a source doc.

---

## 3) Evaluation plan

### 3.1 Headline claim definition

**"We beat BGE-M3 on Kazakh" = we exceed BGE-M3 on KazakhTextDuplicates v2.0 Retrieval MRR (primary headline), with Recall@1 and STS Spearman also >= BGE-M3, AND we do not lose on KazQAD nDCG@10.** Concretely, all of:
- KazakhTextDuplicates MRR **> 0.806** (target a clear gap, ~>=0.83, or significance across >=3 seeds).
- KazakhTextDuplicates Recall@1 **> 0.651**.
- KazakhTextDuplicates STS Spearman **> 0.538 [verify bar]**.
- KazQAD nDCG@10 **> BGE-M3-as-measured-by-us** (we compute BGE-M3's KazQAD number ourselves; the paper's 0.389 is a different retriever).
- Non-regression on MTEB Kazakh (`BelebeleRetrieval` kaz, `SIB200Classification` kaz) vs BGE-M3.

Primary single headline metric if forced to pick one: **KazakhTextDuplicates v2.0 Retrieval MRR.** Secondary headline: **KazQAD nDCG@10** (larger, more realistic corpus).

### 3.2 Baselines to run ourselves (apples-to-apples, one harness, same qrels/prompts)

| Model | HF id | Prompt handling |
|---|---|---|
| BGE-M3 (reference) | `BAAI/bge-m3` | dense; optional query instruction; cosine on L2-normalized |
| multilingual-e5-large | `intfloat/multilingual-e5-large` | **must** prefix `query: ` / `passage: ` (via `get_model`) |
| LaBSE | `sentence-transformers/LaBSE` | no prefix; cosine |
| (secondary) multilingual-e5-instruct | `intfloat/multilingual-e5-large-instruct` | instruction prompt |
| (secondary) GTE-multilingual-base | `Alibaba-NLP/gte-multilingual-base` | no/query prompt |
| (secondary) Snowflake Arctic-Embed v2.0 | `Snowflake/snowflake-arctic-embed-l-v2.0` | query prompt |
| Our model | local checkpoint | our convention |

Run every model through the SAME custom mteb tasks so numbers are directly comparable. First validate the harness by reproducing the paper's BGE-M3 v2.0 MRR 0.806 / Recall@1 0.651 within tolerance; if we can't, fix the protocol before trusting deltas.

### 3.3 Results table template

Kazakh-native (headline):

| Model | KTD MRR | KTD R@1 | KTD R@5 | KTD R@10 | KTD STS ρ | KazQAD nDCG@10 | KazQAD MRR | KazQAD R@100 |
|---|---|---|---|---|---|---|---|---|
| BGE-M3 (paper) | 0.806 | 0.651 | — | — | 0.538[v] | — | — | — |
| BGE-M3 (ours) | | | | | | | | |
| mE5-large (ours) | | | | | | | | |
| LaBSE (ours) | | | | | | | | |
| **KazEmbed (ours)** | | | | | | | | |
| Δ vs BGE-M3(ours) | | | | | | | | |

MTEB Kazakh cross-check + Comparative-QA:

| Model | Belebele-kaz nDCG@10 | WebFAQ-kaz nDCG@10 | SIB200-kaz acc | SIB200clust-kaz v-meas | informatics_kaz Acc@1 | MRR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| BGE-M3 | | | | | | | |
| **KazEmbed (ours)** | | | | | | | |

Report: mean +/- std over >=3 seeds for our model; mark statistically significant wins (paired bootstrap on per-query scores, e.g. p<0.05) — essential given the tiny margins on v2.0.

### 3.4 Order of operations

1. Stand up mteb harness; register KazakhTextDuplicates (STS+Retrieval+PairClf) and KazQAD (Retrieval) as custom tasks; pin dataset revisions.
2. Run BGE-M3 → reproduce paper numbers (KTD) and establish our KazQAD/BelebeleRetrieval BGE-M3 baseline.
3. Run e5-large, LaBSE (+ secondary) through identical harness.
4. Slot in our checkpoints; compute deltas + significance.
5. Final: also run the fresh in-house held-out set to check for memorization.

---

## 4) Pitfalls

1. **Train/test contamination.**
   - KazakhTextDuplicates is generated from a Kazakh source corpus; KazQAD from Kazakh Wikipedia + UNT exams + MT-translated NQ. If our training data overlaps these, results are inflated.
   - Mitigate: exact + MinHash/LSH dedup of training data vs all eval texts and their source docs (Kazakh Wikipedia dump especially, since KazQAD corpus = Wikipedia). Hold out KazQAD `test`, all of KazakhTextDuplicates v2.0, MTEB kaz subsets. Do model selection on a separate dev split, never on the reported test sets. The KazQAD translated-NQ (~61k) is training-only by design — keep it out of test.

2. **Tokenizer / script issues (Cyrillic now, Latin later).**
   - Kazakh Cyrillic uses extra letters: ә ғ қ ң ө ұ ү һ і. Verify the tokenizer/model actually covers them (bge-m3/XLM-R sentencepiece generally do; a from-scratch or English-centric tokenizer may fragment or drop them → check char coverage + avg tokens/word on Kazakh text).
   - **Upcoming Latin script**: Kazakhstan's Cyrillic→Latin reform (roll-out this decade). All current benchmarks are `kaz_Cyrl`. Plan: (a) evaluate primarily on Cyrillic to match the benchmarks; (b) build a transliterated Latin eval slice to measure robustness; (c) consider Cyrillic<->Latin transliteration augmentation in training and a normalization step so the model isn't script-brittle. Report both scripts separately.
   - Mixed Kazakh/Russian code-switching is common — ensure tokenizer handles both.

3. **Query/passage prompt prefixes (asymmetry).**
   - **e5 REQUIRES** `query: ` on queries and `passage: ` on documents (and for STS, symmetric `query: `). Comparing e5 without prefixes badly understates it → unfair "win." Use `mteb.get_model` so prompts are applied automatically.
   - **BGE-M3**: dense mode works without a prefix but supports an optional query instruction; be consistent.
   - **Our model**: define and document a prompt convention; register it in mteb (ModelMeta/model_prompts) so `prompt_type` (query vs passage) is applied identically in every task. Never compare a prefixed model to a non-prefixed one on the same axis without noting it.

4. **Normalization & similarity function.**
   - Cosine similarity requires **L2-normalized** embeddings. bge-m3 dense + e5 + LaBSE all expect cosine on normalized vectors. Set `normalize_embeddings=True` (or rely on mteb applying each model's declared `similarity_fn`). A mismatch (dot vs cosine, unnormalized) silently changes rankings — keep it consistent across all models and matched to each model's recommendation.
   - Watch pooling (mean vs CLS): bge-m3 = CLS/dense head; e5 = mean pooling; LaBSE = CLS. `sentence-transformers`/mteb handle this per model — do not hand-roll pooling for baselines.

5. **Tiny-margin / ranking-first trap.**
   - v2.0 top models are within ~0.003 MRR. A raw 0.001 improvement is meaningless. Require a real gap and/or paired-bootstrap significance over multiple seeds. Report Recall@5/@10 too (they saturate — good for sanity, poor for differentiation), and lead with Recall@1/MRR where the signal is.

6. **Harness fidelity.**
   - Reproduce the published BGE-M3 numbers before claiming any delta; a mismatch means our candidate pool / scoring / normalization differs from the paper, not that we won. Pin dataset revisions and record mteb version (2.21.0), model revisions, and encode_kwargs in every result JSON.

---

## Key repos / IDs (quick reference)

- KazakhTextDuplicates: HF `Arailym-tleubayeva/KazakhTextDuplicates`; paper https://www.mdpi.com/2306-5729/11/6/133
- KazQAD: HF `issai/kazqad`, `issai/kazqad-retrieval`; GitHub https://github.com/IS2AI/KazQAD; arXiv 2404.04487
- Comparative-QA dataset: HF `Kundyzka/informatics_kaz`; paper ResearchGate 396213005; related repo https://github.com/Arailym-ray/KAZ-QA-RAG
- MTEB: https://github.com/embeddings-benchmark/mteb (v2.21.0); MMTEB arXiv 2502.13595
- TUMLU (out of scope for embeddings): arXiv 2502.11020; KazMMLU arXiv 2502.12829
- Baselines: `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `sentence-transformers/LaBSE`, `Alibaba-NLP/gte-multilingual-base`, `Snowflake/snowflake-arctic-embed-l-v2.0`
