# Kazakh (kk) Embedding Model — Data Source Inventory

**Goal:** Build a Kazakh text embedding model that beats **BGE-M3** on Kazakh benchmarks, via (a) continued pretraining and (b) contrastive fine-tuning.
**Compiled:** 2026-09-20 · **Method:** web search (EN/RU/KK), Hugging Face, OPUS API, GitHub. Repo IDs verified where possible; unverified ones marked **verify**.

Local generation assets available: **`issai/LLama-3.1-KazLLM-1.0-70B`** (CC BY-NC 4.0) + **1× RTX 3090** — usable for synthetic query/pair generation and translation.

Key baselines to beat / reuse: **`BAAI/bge-m3`** (target), **`intfloat/multilingual-e5-large`**, **`sentence-transformers/LaBSE`**, and the existing Kazakh model **KazEmbed-v5** (E5-based; **verify** exact HF repo id).

---

## TL;DR — TOP 10 highest-value sources

| # | Source | HF repo ID / URL | Use | Why it's top |
|---|--------|------------------|-----|--------------|
| 1 | **KazQAD** (+ retrieval split) | `issai/kazqad`, `issai/kazqad-retrieval` | Contrastive positive pairs (query→passage) + **eval** | Only native Kazakh open-domain QA/retrieval set; direct in-domain training + eval signal for beating BGE-M3 on retrieval. Gated. |
| 2 | **KazParC** | `issai/kazparc` | Contrastive pairs (translation = cross-lingual positives) | 372k human + ~1.8M synthetic kk↔en/ru/tr pairs; gold for multilingual alignment. CC BY 4.0 (gated on HF). |
| 3 | **OPUS NLLB en-kk** | via `allenai/nllb` / OPUS `object.pouta.csc.fi/OPUS-NLLB` | Cross-lingual positive pairs | ~34.6M mined en-kk pairs — by far the largest parallel pool. Noisy → filter by LaBSE/margin score. |
| 4 | **CulturaX (kk)** | `uonlp/CulturaX` (config `kk`) | Continued pretraining + pair mining | 2.73M docs / 2.8B tokens, cleaned+deduped. Gated (CC-ish, per-source). Highest-quality single web corpus. |
| 5 | **FineWeb-2 (kaz_Cyrl)** | `HuggingFaceFW/fineweb-2` (config `kaz_Cyrl`) | Continued pretraining | 3.38M rows / 6.86 GB, strong quality filtering; complements HPLT. ODC-BY. |
| 6 | **HPLT 3.0 (kk)** | `HPLT/HPLT3.0` | Continued pretraining | ~7.34B tokens; Internet-Archive crawls complement CommonCrawl-only sets. Open (CC0-ish per-doc). |
| 7 | **KazakhTextDuplicates v2.0** | `Arailym-tleubayeva/KazakhTextDuplicates` | **Eval** (STS/dedup/retrieval) + paraphrase positives | Purpose-built Kazakh semantic-similarity/retrieval benchmark; BGE-M3 already scored on it → direct comparison target. Open. |
| 8 | **MDBKD** (multidomain) | `kz-transformers/multidomain-kazakh-dataset` | Continued pretraining | 24.9M texts / 2.09B tokens; Apache-2.0 (fully open); bundles CC100+wiki+books+Leipzig+OSCAR+news. |
| 9 | **Til-Corpus** | `TilQazyna/Til-Corpus` | Continued pretraining (largest) | 204 GB / 58.8M docs, quality-tiered, KK-centric. Gated — request access. Single biggest KK text pool. |
| 10 | **Kardeş-NLU (kk)** | GitHub `lksenel/Kardes-NLU` (HF **verify**) | NLI + STS positives/eval | Human-post-edited Kazakh XNLI + STS-B + XCOPA — the only real KK NLI/STS; NLI triplets are premium contrastive fuel. |

**Estimated total accessible volume:** ~**300+ GB** raw KK-centric text (Til-Corpus dominates); realistically ~**10–20 GB / ~8–15B unique tokens** of high-quality deduplicated monolingual text after cross-source dedup (CulturaX∪FineWeb2∪HPLT∪MDBKD). Parallel/paired: ~**36M** cross-lingual pairs (mostly NLLB, noisy) + ~**0.9M** gold human pairs + ~**0.9M** native QA/instruction pairs.

---

## 1) Monolingual Kazakh corpora (continued pretraining + pair mining)

| Source | HF repo ID / URL | Size | Rows/Tokens | License | Domain | Note |
|--------|------------------|------|-------------|---------|--------|------|
| **CulturaX (kk)** | `uonlp/CulturaX` cfg `kk` | 9.07 GB | 2.73M docs / 2.80B tok | Gated; per-source (mC4+OSCAR terms) | Web crawl | Pretraining; top-quality dedup. Also mine intra-doc sentence pairs. |
| **FineWeb-2 (kaz_Cyrl)** | `HuggingFaceFW/fineweb-2` cfg `kaz_Cyrl` | 6.86 GB | 3.38M rows | ODC-BY | Web crawl | Pretraining; strong filtering; complements HPLT (both CC). |
| **HPLT 3.0 (kk)** | `HPLT/HPLT3.0` | 8.76 GB | ~5.12M docs / ~7.34B tok | Open (per-doc CC0) | Web + Internet Archive | Pretraining; IA crawls add non-CC coverage. (v2 also: `HPLT/HPLT2.0_cleaned`.) |
| **MDBKD** (Multi-Domain Bilingual Kazakh) | `kz-transformers/multidomain-kazakh-dataset` | 24.7 GB | 24.9M texts / 2.09B tok | **Apache-2.0** | Wiki+Books+Leipzig+OSCAR+News | Pretraining; fully open, ready-mixed 5 splits. Best "safe-license" bulk. |
| **Til-Corpus** | `TilQazyna/Til-Corpus` | **204.2 GB** | 58.85M docs | Gated | KK-centric mixed (kk/ru/en) | Pretraining; largest single KK pool; quality-tiered. Request access. |
| **Til-Parallel** | `TilQazyna/Til-Parallel` | 72.16 GB | 23.29M rows | Gated | Multilingual, KK-centered | Pretraining + pair mining. |
| **Til-Books** | `TilQazyna/Til-Books` | 2.02 GB | 16,847 books (13,194 kk) | Gated | Books (OCR + clean) | Pretraining; long-form clean text. |
| **kaz-text-for-lm-normalized** | `farabi-lab/kaz-text-for-lm-normalized` | 5.99 GB | n/r | Gated | News+lit+academic+Wiki(2024-08) | Pretraining; already normalized. |
| **CC-100 (kk)** | `statmt/cc100` cfg `kk` | 889 MB | n/r | Open (CC / CommonCrawl terms) | Web crawl | Pretraining; classic, smaller. |
| **mOSCAR (kk)** | `oscar-corpus/mOSCAR` cfg `kaz_Cyrl` | 548.6 MB | 248,403 docs | Open (per-source) | Web (multimodal release, text usable) | Pretraining; modest size. |
| **OSCAR (kk)** | `oscar-corpus/OSCAR-2301` (cfg `kk`), `oscar-corpus/OSCAR-2201`, `-2109` | ~n/r | n/r | Gated (agree to terms) | Web crawl | Pretraining; superseded by CulturaX/FineWeb but useful for dedup diversity. |
| **Wikipedia (kk)** | `wikimedia/wikipedia` cfg `20231101.kk`; also `amandyk/kazakh_wiki_articles` | ~0.4–0.5 GB | ~240k articles | CC BY-SA 4.0 | Encyclopedic | Pretraining + clean pair source (title↔section, intra-article). |
| **Leipzig Corpora (kaz)** | `imvladikon/leipzig_corpora_collection`; site: wortschatz.uni-leipzig.de/en/download/Kazakh (`kaz-kz_web_2019`) | ~n/r | 17.06M sentences / 209.5M tok | CC BY-NC 4.0 | Web/news sentences | Pretraining + sentence bank for synthetic pairs. Non-commercial. |
| **mC4 (kk)** | `allenai/c4` cfg `multilingual` (kk split); legacy `mc4` | large | n/r | ODC-BY | Web crawl | Pretraining; largely subsumed by CulturaX/FineWeb. |
| **KazNewsDataset** | Mendeley `hwj24p9gkh/1` | n/r | 1,142,735 docs | Open | KZ news | Pretraining + in-domain news pairs. |
| **KazRusNewsDataset** | Mendeley `2vz7vtbhn2/1` | n/r | 6,261,953 docs | Open | KZ+RU news | Pretraining; large but mixed-lang. |
| **Textual Foundations of Justice** | Mendeley `jdpc5658nh/3` | n/r | all RK laws (kk+ru) | Open | Legal | Domain pretraining; legal retrieval pairs. |
| **MADLAD-400 (kk)** | `allenai/MADLAD-400` cfg `kk` (**verify size**) | large | n/r | ODC-BY | Web crawl | Pretraining; extra dedup diversity. |

**Caveats:** CulturaX/FineWeb-2/OSCAR/mC4 are all CommonCrawl-derived → heavy overlap; **dedupe across them** (MinHash) before use. HPLT adds Internet-Archive coverage → keep it. Gated repos (Til-*, CulturaX, OSCAR, KazParC, KazQAD, KazSAnDRA, KazNERD) require clicking access / login token. Til-Corpus license terms unstated beyond "gated" — **verify commercial use**.

---

## 2) Paired / labeled data for contrastive training

### 2a. Native Kazakh QA / retrieval / paraphrase (highest signal)

| Source | HF repo ID / URL | Size | License | Use | Note |
|--------|------------------|------|---------|-----|------|
| **KazQAD** | `issai/kazqad` | 281.7 MB | Gated (CC BY-SA-ish, **verify**) | (query, gold passage) positives + reading comp | ~6k questions, 12.7k passages; from translated NQ + Kazakh UNT. |
| **KazQAD-retrieval** | `issai/kazqad-retrieval` | — | Gated | Retrieval training + **eval**; ~800k Wiki passages corpus | The passage corpus + qrels: ideal hard-negative mining pool. |
| **KazakhTextDuplicates v2.0** | `Arailym-tleubayeva/KazakhTextDuplicates` | 217 MB | Open | Paraphrase/near-dup positives + **eval** | 25,922 rows, 7 graded duplication regimes (paraphrase, partial, noise). |
| **Kazakh Open Retrieval Benchmark** | `Tim2190/kaz-rag-search-benchmark` | 14.6 MB | Open | Retrieval **eval** + pairs | 300 queries / 8,370 passages from KK Wikipedia. |
| **Zerde-QA-50K** | `kurumikz/Zerde-QA-50K` | 156.2 MB | Open | Synthetic QA positives | 51,422 KK QA pairs, 20+ domains; LLM-generated (check quality). |
| **KazCulture** | `issai/KazCulture` | 4.6 MB | Gated | (passage, Q, A) triplets → positives + eval | 16,137 human PQA triplets on KK culture. |
| **RAGBench Kazakh** | `issai/RAGBench_Kazakh` | 31.2 MB | Open | RAG query→context positives + eval | 11,431; translated RAGBench (biomed/legal/finance/general). |
| **Kazakh Analytical RAG** | `farabi-lab/KZ-RAG-single-docs-final-gold` | 33.4 MB | Gated | (question, doc) positives | 4,522 examples, single-doc context. |
| **Multi-Step Reasoning KK** | `farabi-lab/multi_step_reasoning_kazakh_context` | 39.0 MB | Gated | QA positives | 10,981 examples, multi-hop. |
| **Kazakh Instruction v2** | `AmanMussa/kazakh-instruction-v2` | 35.6 MB | Open | Instruction pairs (weak positives) | 52,201 rows; translated Alpaca + KZ additions. |
| **Kazakh-IFT** | `nurkhan5l/kazakh-ift` | 7.85 MB | Gated | Instruction pairs | ~10,600; GPT-4o-generated, governance/legal/culture. |

### 2b. NLI / STS (premium for contrastive; triplets → hard negatives)

| Source | Repo / URL | Size | License | Use | Note |
|--------|-----------|------|---------|-----|------|
| **Kardeş-NLU (kk)** | GitHub `lksenel/Kardes-NLU` (HF **verify**) | small | research (**verify**) | NLI triplets + STS-B pairs + XCOPA | Human-post-edited KK XNLI/STS/COPA. Only real native KK NLI/STS. Gold for (anchor, entail+, contradict−). |
| **KazakhTextDuplicates** | (see 2a) | 217 MB | Open | Graded STS pairs | Doubles as STS training + eval. |

*No large native KK STS-benchmark beyond the two above → plan to **translate** English NLI (SNLI/MNLI/AllNLI) and STS-B into KK with KazLLM (see §4).*

### 2c. Parallel corpora (translation = cross-lingual positive pairs; "gold for multilingual alignment")

Human / curated:

| Source | HF repo ID / URL | Pairs | License | Pair | Note |
|--------|------------------|-------|---------|------|------|
| **KazParC** | `issai/kazparc` | 372k human (+~1.8M SynC) | CC BY 4.0 (gated) | kk↔en/ru/tr | Best curated multi-domain KK parallel. |
| **Kazakh-English KAZNU** | `Dauren-Nur/kaz_eng_parallel` | 377,044 | Open | kk↔en | Law + news. |
| **Kazakh-Russian KAZNU** | `Dauren-Nur/kaz_rus_parallel_corpora_KAZNU` | 86,453 | Open | kk↔ru | Governmental/official docs. |
| **KazLit-Parallel** | `SagiAbd/KazLit-Parallel` | 71,096 | Open | kk/ru/en | Literary; auto-aligned (some noise). |
| **Uzbek-Kazakh Parallel** | `Sanatbek/uzbek-kazakh-parallel-corpora` | 133,877 | Open | uz↔kk | Expert-translated; Turkic transfer. |
| **KazParaD** | `issai/KazParaD` (**verify contents**) | n/r | Gated | kk↔… | ISSAI paraphrase/parallel dataset — verify. |
| **FLORES-200 (kk)** | `facebook/flores` cfg `kaz_Cyrl` | 3,001 | CC BY-SA 4.0 | multi | **Eval** for translation/bitext alignment. |

Web-mined via **OPUS** (verified via OPUS API, latest versions). Download: `object.pouta.csc.fi/OPUS-<corpus>/<ver>/moses/<pair>.txt.zip`.

**kk↔en (OPUS):**

| Corpus | Pairs | Note |
|--------|-------|------|
| **NLLB** | **34,589,761** | Dominant; noisy web-mined — filter by LASER/LaBSE margin. Also on `allenai/nllb`. |
| CCAligned | 689,653 | Web-mined; moderate noise. |
| KDE4 | 69,833 | Software UI. |
| OpenSubtitles | 61,649 | Subtitles (short, colloquial). |
| GNOME | 20,978 | Software UI. |
| WikiMatrix | 20,054 | Wikipedia-mined, cleaner. |
| TED2020 | 9,639 | Talks. |
| News-Commentary | 9,871 | News. |
| WMT-News | 6,130 | News. |
| QED | 5,275 | Educational. |
| NeuLab-TedTalks | 3,727 | Talks. |
| ELRC-Kazakh_Legal_MT | 1,002 | Legal. |
| Tatoeba | 466 | Tiny, high-quality sentences. |

**kk↔ru (OPUS):**

| Corpus | Pairs | Note |
|--------|-------|------|
| MultiCCAligned | 431,953 | Largest kk-ru; web-mined. |
| XLEnt | 87,168 | Entity-centric mined. |
| wikimedia | 64,480 | Wikipedia. |
| KDE4 | 68,014 | Software UI. |
| OpenSubtitles | 54,475 | Subtitles. |
| WikiMatrix | 32,808 | Wikipedia-mined. |
| GNOME | 20,550 | Software UI. |
| TED2020 | 9,610 | Talks. |
| News-Commentary | 9,437 | News. |
| QED | 5,125 | Educational. |
| NeuLab-TedTalks | 3,150 | Talks. |
| Tatoeba | 2,431 | High-quality. |

*(For kk↔ru, NLLB/CCMatrix also exist via `allenai/nllb`; OPUS API returned them primarily on the en side. **Verify** kk-ru NLLB volume directly if needed.)*
Aggregate hubs: **`allenai/nllb`** (all NLLB pairs), OPUS portal `https://opus.nlpl.eu/`. **Caveat:** web-mined pairs need margin-score filtering (LaBSE cosine > ~0.7) before use as positives.

---

## 3) Evaluation datasets specific to Kazakh

| Source | HF repo ID / URL | Task type | Size | License | Note |
|--------|------------------|-----------|------|---------|------|
| **KazakhTextDuplicates v2.0** | `Arailym-tleubayeva/KazakhTextDuplicates` | STS / dedup / retrieval | 25,922 | Open | **Primary target**: BGE-M3, LaBSE, mE5-large, KazEmbed-v5, text-embedding-3-large already benchmarked here (MDPI 2306-5729/11/6/133). |
| **KazQAD retrieval** | `issai/kazqad-retrieval` | Retrieval (nDCG/Recall) | ~800k passages | Gated | Native ODQA retrieval eval. |
| **Kazakh Open Retrieval Benchmark** | `Tim2190/kaz-rag-search-benchmark` | Retrieval | 300 q / 8,370 p | Open | Evidence-based; shows stemming beats multiling. embeds → clear bar to beat. |
| **TUMLU-mini** | `jafarisbarov/TUMLU-mini` | Classification/MCQA (KK subset) | part of 38,139 | Open (**verify**) | Native Turkic understanding; KK subset for classification eval. |
| **KazSAnDRA** | `issai/kazsandra` | Sentiment classification | 180,064 | Gated (CC BY 4.0) | MTEB-style classification eval + training. |
| **KazNERD** | `issai/kaznerd` (also `yeshpanovrustem/ner-kazakh`) | NER | 112,702 sent | Gated (CC BY 4.0) | Token-level; less direct for sentence-embeds but usable for clustering eval. |
| **Kardeş-NLU (kk)** | GitHub `lksenel/Kardes-NLU` | NLI / STS / COPA | small | research | STS pair eval (Spearman). |
| **SIB-200 (kk)** | `Davlan/sib200` cfg `kaz_Cyrl` | Topic classification | 1,004 | Open (CC BY-SA) | In MTEB/MMTEB as SIB200Classification. |
| **KazMMLU** | `MBZUAI/KazMMLU` | MCQA (knowledge) | 10,969 kk | Open | Knowledge eval (for retrieval-augmented framing). |
| **KazBench-KK** | `kz-transformers/kk-socio-cultural-bench-mc` | Cultural MCQA | 7,111 | Open | Cultural grounding eval. |
| **FLORES-200 (kk)** | `facebook/flores` cfg `kaz_Cyrl` | Bitext mining eval | 3,001 | CC BY-SA | Cross-lingual alignment (kk↔en/ru) eval. |
| **MTEB / MMTEB (kk coverage)** | GitHub `embeddings-benchmark/mteb` | Aggregated | — | mixed | KK appears via SIB-200 (classification) + Belebele-derived retrieval + FLORES bitext; **verify** exact task list (`mteb` `--languages kaz`). No large native KK STS/retrieval in core MTEB yet → use KazakhTextDuplicates + KazQAD as primary. |

---

## 4) Synthetic data strategies (leverage local KazLLM-70B + RTX 3090)

Standard low-resource embedding recipes (mE5, "Less is More" arXiv 2603.22290, Armenian/Turkish TurkEmbed, Kazakh skill-matching Frontiers 2026.1900836) all rely on synthetic + translated data. Concrete plays:

| Strategy | How | Source data | Output | Caveat |
|----------|-----|-------------|--------|--------|
| **LLM query generation** | Prompt KazLLM-70B: "write N questions answered by this passage" over KK passages | CulturaX/FineWeb2/HPLT/Wikipedia/KazQAD passages | (synthetic query, gold passage) positives at scale | Filter for answerability; dedup; mix with in-batch + mined hard negatives. |
| **Translate EN retrieval sets** | MT EN→KK with KazLLM-70B or `issai/tilmash` | MS MARCO / mMARCO, NQ, HotpotQA, MLDR | Large KK (query, passage) pairs | Translation drift; keep EN passage too for cross-lingual anchor. mMARCO already multilingual. |
| **Translate NLI/STS** | MT EN→KK | SNLI/MNLI/AllNLI, STS-B | KK NLI triplets (anchor, +, −) + STS pairs | Best cheap source of hard negatives; post-filter with KazLLM as judge. |
| **Back-translation paraphrase** | kk→ru/en→kk round-trip (Tilmash/NLLB) | KK monolingual sentences | Paraphrase positive pairs | Cheap positives; watch semantic drift. Complements KazakhTextDuplicates regimes. |
| **Parallel as positives** | Use §2c pairs directly | KazParC, NLLB, OPUS | Cross-lingual positives (kk↔en/ru) | Filter web-mined by LaBSE margin > 0.7. |
| **Hard-negative mining** | Retrieve top-k with a base encoder (BGE-M3/mE5), take rank 10–100 as negatives | KazQAD/Wikipedia corpus | (q, pos, hardneg) triplets | Essential to beat BGE-M3; distill from cross-encoder if compute allows. |
| **Cross-encoder distillation** | Score (q,d) with KazLLM-70B or bge-reranker, distill into bi-encoder | any pairs | soft labels | mE5 recipe; boosts quality most. RTX 3090 fine for bi-encoder train (LoRA / small batch + grad-cache). |

Generation tools on HF: `issai/tilmash` (kk↔en/ru/tr MT), `issai/LLama-3.1-KazLLM-1.0-70B` (use quantized `...-AWQ4`/`-GGUF4` to fit generation throughput; 70B won't fit one 3090 unquantized — run generation via CPU-offload/quant or the 8B `issai/LLama-3.1-KazLLM-1.0-8B` for volume).

---

## Recommended data plan (summary)

- **Continued pretraining pool:** MDBKD (Apache-2.0, safe) + FineWeb-2 kaz_Cyrl + HPLT 3.0 + CulturaX, cross-deduped → ~8–15B unique tokens. Add Til-Corpus if access + license clear.
- **Contrastive fine-tuning positives:** KazQAD + KazParC + filtered OPUS/NLLB (kk-en, kk-ru) + Kardeş-NLU NLI triplets + KazakhTextDuplicates paraphrases + synthetic LLM-generated (query,passage) + translated mMARCO/AllNLI.
- **Hard negatives:** mine from KazQAD/Wikipedia passage corpus with a base encoder; optionally distill from bge-reranker/KazLLM.
- **Eval (must-beat-BGE-M3):** KazakhTextDuplicates v2.0 (STS/retrieval), KazQAD-retrieval, Kazakh Open Retrieval Benchmark, FLORES bitext, SIB-200 + KazSAnDRA classification.

## Gaps / risks
- **No large native KK STS or MS-MARCO-scale KK retrieval train set** → must synthesize/translate. Biggest gap.
- Many best sources are **gated** (Til-*, CulturaX, OSCAR, KazParC, KazQAD, KazSAnDRA, KazNERD) — provision HF token + accept terms; **verify commercial licenses** (KazLLM CC BY-NC = non-commercial outputs; Leipzig CC BY-NC).
- Web-mined parallel (NLLB/CCAligned/MultiCCAligned) is **noisy** — mandatory margin filtering.
- **Verify** flags: Kardeş-NLU HF path, KazEmbed-v5 repo id, KazParaD contents, MADLAD-400 kk size, exact MTEB kk task list, kk-ru NLLB volume.

## Source links
- Catalog: https://github.com/Allessyer/awesome-kaz-datasets (verified 2026-08-21) · https://github.com/salyamq/kaz-datasets
- ISSAI datasets: https://issai.nu.edu.kz/issai-datasets/ · https://github.com/IS2AI
- OPUS: https://opus.nlpl.eu/ (API: `opus.nlpl.eu/opusapi/?source=kk&target=en`)
- KazParC paper: https://arxiv.org/abs/2403.19399 · KazQAD: https://arxiv.org/abs/2404.04487
- KazakhTextDuplicates: https://www.mdpi.com/2306-5729/11/6/133
- TUMLU: https://arxiv.org/abs/2502.11020 · Kardeş-NLU: https://aclanthology.org/2024.eacl-long.100/
- mE5 report: https://arxiv.org/abs/2402.05672 · Low-resource embeds: https://arxiv.org/html/2603.22290
