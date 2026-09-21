#!/usr/bin/env python3
"""Скоринг триплетов кросс-энкодером-учителем (bge-reranker-v2-m3) для дистилляции.
Разворачивает n-tuple (anchor, positive, negative_1..k) в строки (anchor, positive,
negative) и пишет label = teacher_score(a,pos) - teacher_score(a,neg) — маржа для MarginMSE.

    python scripts/06_score_teacher.py --in data/pairs/train_triplets.jsonl \
        --out data/pairs/train_distill.jsonl --reranker BAAI/bge-reranker-v2-m3 --max-neg 4
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pairs/train_triplets.jsonl")
    ap.add_argument("--out", default="data/pairs/train_distill.jsonl")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-neg", type=int, default=4)
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import CrossEncoder

    ds = load_dataset("json", data_files={"t": args.inp}, split="t")
    negcols = [c for c in ds.column_names if c.startswith("negative")][:args.max_neg]
    anchors, poss, negs = [], [], []
    for r in ds:
        for nc in negcols:
            n = r.get(nc)
            if r["anchor"] and r["positive"] and n:
                anchors.append(r["anchor"]); poss.append(r["positive"]); negs.append(n)
    print(f"строк для скоринга: {len(anchors)} (из {len(ds)} триплетов × {len(negcols)} нег.)")

    ce = CrossEncoder(args.reranker, max_length=512)
    sp = ce.predict(list(zip(anchors, poss)), batch_size=64, show_progress_bar=True)
    sn = ce.predict(list(zip(anchors, negs)), batch_size=64, show_progress_bar=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for a, p, n, s1, s2 in zip(anchors, poss, negs, sp, sn):
            f.write(json.dumps({"anchor": a, "positive": p, "negative": n,
                                "label": float(s1) - float(s2)}, ensure_ascii=False) + "\n")
    print(f"записано {len(anchors)} строк -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
