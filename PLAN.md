# Казахская эмбеддинг-модель, обходящая BGE-M3 — мастер-план

Синтез 4 исследовательских отчётов (`research/01..04`). Цель: обучить казахскую
text-embedding модель, которая **бьёт BGE-M3** на казахских бенчмарках, собрав
максимум данных из интернета и расширив корпус для дообучения.
Железо: `debian3090` (RTX 3090, 24 ГБ; диск 1.4 ТБ свободно).

---

## 1. Цель и планка (что именно «обойти»)

| Бенчмарк | Метрика | BGE-M3 (побить) | Заметка |
|----------|---------|-----------------|---------|
| **KazQAD retrieval** (PRIMARY) | nDCG@10 / R@1 / MRR | R@1 **0.641** / R@10 **0.929** / MRR **0.746** | главный таргет; свой замер BGE-M3 обязателен |
| **KazakhTextDuplicates** STS | Spearman | **0.538** | вторичный |
| KazakhTextDuplicates retrieval | MRR | 0.806 | ⚠️ насыщен (все в ±0.003) — как цель слаб |
| MMTEB kk-подмножества | nDCG@10 / acc | — | non-regression (Belebele-kaz, SIB200-kaz) |

**Условие победы:** превзойти BGE-M3 на KazQAD (nDCG@10 и R@1) с реальным зазором
(не 0.001), при отсутствии регресса на STS и MMTEB-kk. Замер: ≥3 сида + paired-bootstrap.

---

## 2. База для дообучения

- **PRIMARY: `Alibaba-NLP/gte-multilingual-base`** — 305M, dim 768 (Matryoshka),
  ctx 8192, Apache-2.0. Единственная компактная модель с подтверждённым тяжёлым
  казахским претрейном (1.02B kk токенов), уже выигрывала в informatics_kaz.
  Влезает в **full fine-tune с большим батчем** на 3090 — это главный рычаг.
- **SECONDARY: `Snowflake/snowflake-arctic-embed-l-v2.0`** — 568M, dim 1024→256,
  Apache-2.0, #2 на KazQAD. Full FT c grad-checkpointing или LoRA.
- **FALLBACK / TEACHER: `BAAI/bge-m3`** (MIT) — дообучить на казахском; почти
  гарантированно бьёт zero-shot BGE-M3; и хороший teacher для дистилляции.
- **Avoid:** jina-v3 (CC-BY-NC), paraphrase-multilingual-* (нет kk), Qwen3-4B
  (только QLoRA на 3090 → маленький батч вредит контрастиву), LaBSE (слабый retriever).

Стратегия: тренируем gte (primary) и bge-m3 (fallback), оставляем победителя.

---

## 3. Данные — расширение базы (см. `data/datasets.yaml`)

**Претрейн-корпус (моно-kk):** CulturaX kk, FineWeb-2 kaz_Cyrl, HPLT 3.0 kk,
MDBKD (Apache-2.0, безопасная лицензия), Til-Corpus (204 ГБ). После дедупа
~8–15B уникальных токенов.

**Пары для контрастива (золото):** KazQAD (нативные), KazParC (372k чел.),
Kardeş-NLU kk (NLI/STS), OPUS kk-ru/kk-en (CCAligned/WikiMatrix/Tatoeba/OpenSubtitles).

**Синтетика (закрывает главный пробел — нет большого нативного KK STS/MS-MARCO):**
1. перевод mMARCO / AllNLI / STS-B в казахский через `issai/tilmash`;
2. генерация запросов из kk-пассажей локальной **KazLLM** (8B/AWQ на 3090);
3. back-translation парафразы;
4. фильтрация всех пар: LaBSE cosine > 0.7 + round-trip.

**Лицензии:** для коммерции — MDBKD (Apache), KazParC (CC-BY), KazQAD (CC-BY-SA).
KazLLM и Leipzig = CC-BY-**NC** (только research). ← нужно решение о цели проекта.

---

## 4. Методология обучения (под 3090)

1. Data prep → пул пар из 4 источников (нативные + переводные + параллельные + синтетика), LaBSE-фильтр.
2. (опц.) Continued pretraining (RetroMAE/MLM) на моно-kk — только если база слаба на kk.
3. **Hard-negative mining** (NV-Retriever: top, relative_margin 0.05, max_score ~0.8, 5–15 негативов, faiss) — **рычаг №1**.
4. **Дистилляция** из кросс-энкодера `BAAI/bge-reranker-v2-m3` (MarginMSE/listwise) + контрастивный член.
5. **Контрастив:** `CachedMultipleNegativesRankingLoss` (или CachedGISTEmbedLoss, guide=bge-m3) в обёртке `MatryoshkaLoss([768,512,256,128,64])`; bf16 + gradient_checkpointing + 8-bit Adam + GradCache; mini-batch 16 → эффективный 512–1024; temp 0.02.
6. **Eval каждую итерацию** vs BGE-M3 на KazQAD + MMTEB-kk.
7. **Итерации:** re-mine hard-negatives улучшенным чекпоинтом → re-score teacher → +1 эпоха. 2–3 круга.
8. Экспорт Matryoshka-размерностей (768/512/256).

Ключевые рычаги (по ROI): hard-neg + фильтр ложных → объём чистых триплетов →
дистилляция → большой батч → синтетика KazLLM → кросс-язычный якорь kk-ru/kk-en.

**Тайминг:** seq 512, ~1.5–3 it/s, несколько часов–день на прогон, план 2–3 прогона.

---

## 5. Оценка (харнесс)

- `mteb==2.21.0` + sentence-transformers. В mteb **нет** KazQAD/KTD — добавляем как
  custom `AbsTaskRetrieval`/`AbsTaskSTS`.
- Сначала **воспроизвести опубликованные числа BGE-M3** (0.806/0.651, KazQAD) —
  только потом верить дельтам.
- Все бейзлайны (bge-m3, mE5-large, LaBSE) гоняем сами через один харнесс.
- e5 — обязательны префиксы `query:`/`passage:` (через `mteb.get_model`).
- Дедуп train↔eval (MinHash+exact) против тест-сплитов И их источников (kk-Wikipedia!).

---

## 6. Порядок исполнения (что делаем по шагам)

- [ ] **0. Решения пользователя** (см. блокеры ниже): цель (коммерция/research),
      освобождение GPU, HF-токен.
- [ ] 1. Окружение на 3090 (`scripts/00_setup_env.sh` — uv + py3.11 + torch cu12 + st/FlagEmbedding/mteb).
- [ ] 2. Скачать данные (`scripts/01_download_data.py` из `data/datasets.yaml`).
- [ ] 3. Построить eval-харнесс, воспроизвести BGE-M3 бейзлайны.
- [ ] 4. Собрать пары + синтетика (перевод tilmash + генерация KazLLM) + LaBSE-фильтр + дедуп.
- [ ] 5. Hard-neg mining.
- [ ] 6. Train (gte primary, bge-m3 fallback) + Matryoshka + дистилляция.
- [ ] 7. Eval vs BGE-M3, итерации, выбор победителя, экспорт.

---

## 7. Решения пользователя (зафиксировано 2026-09-20)

1. **Цель: Research-only** → разрешены CC-BY-NC источники (синтетика KazLLM, Leipzig, jina).
2. **GPU: останавливать `llama-server` на время обучения** (с разрешения; вернуть после). Для setup/download GPU не трогаем.
3. **Старт: да** — поднять окружение + качать негейтед-данные сейчас; gated — после HF-токена.
4. **База: обе** — `gte-multilingual-base` (primary) + `bge-m3` (fallback), оставить победителя.

## 7b. Блокеры / решения (нужны от пользователя)

1. **GPU занят:** `llama-server` (llama.cpp) держит 20.9/24 ГБ VRAM + почти всю RAM.
   Для обучения нужно его остановить на время / тренировать по расписанию. Свою LLM
   я не трогаю без разрешения.
2. **Цель проекта — коммерция или research?** Определяет, можно ли использовать
   CC-BY-NC данные (KazLLM-синтетика, Leipzig) и non-commercial базы.
3. **HF-токен** нужен для gated-датасетов (KazQAD, CulturaX, Til-Corpus…).
4. Подтвердить приоритет базы: gte-multilingual-base (рекомендуется) vs bge-m3.
