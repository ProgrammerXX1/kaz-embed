#!/usr/bin/env python3
"""Майнинг hard-negatives (рычаг №1 для обхода BGE-M3).
Использует sentence-transformers.util.mine_hard_negatives: строит корпус из
positive-пассажей, ретривит близкие и берёт трудные негативы с фильтром ложных
(NV-Retriever-подобные настройки). Выход: n-tuple (anchor, positive, negative_1..k).

    python scripts/05_mine_negatives.py --model models/bge-m3-st \
        --in data/pairs/train.jsonl --out data/pairs/train_triplets.jsonl \
        --num-neg 4 --limit 60000
"""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bge-m3-st", help="майнер (сильный ретривер)")
    ap.add_argument("--in", dest="inp", default="data/pairs/train.jsonl")
    ap.add_argument("--out", default="data/pairs/train_triplets.jsonl")
    ap.add_argument("--num-neg", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help=">0 — взять первые N пар (скорость)")
    ap.add_argument("--corpus-full", action="store_true", help="майнить негативы из полного KazQAD 825k")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives

    ds = load_dataset("json", data_files={"t": args.inp}, split="t").select_columns(["anchor", "positive"])
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"пар на вход: {len(ds)}")

    model = SentenceTransformer(args.model, trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 256

    corpus = None
    if args.corpus_full:     # майним негативы из ПОЛНОГО корпуса 825k (реальные дистракторы)
        c = load_dataset("issai/kazqad-retrieval", "corpus", split="train")
        corpus = [((t or "") + ". " + (x or "")).strip() for t, x in zip(c["title"], c["text"])]
        print(f"корпус для майнинга: {len(corpus)} пассажей")

    mined = mine_hard_negatives(
        dataset=ds,
        model=model,
        corpus=corpus,           # None => из позитивов; иначе из полного корпуса
        anchor_column_name="anchor",
        positive_column_name="positive",
        num_negatives=args.num_neg,
        range_min=10,            # пропускаем топ-10 (вероятные ложные негативы)
        range_max=60,
        max_score=0.8,           # отсекаем слишком похожие (ложные негативы)
        margin=0.0,
        sampling_strategy="top",
        batch_size=256,
        use_faiss=True,
        output_format="n-tuple",  # anchor, positive, negative_1..k в одной строке
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mined.to_json(args.out, force_ascii=False)
    print(f"триплетов: {len(mined)} cols={mined.column_names} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
