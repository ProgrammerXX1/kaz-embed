#!/usr/bin/env python3
"""Гейт по коллапсу: средняя попарная косинусная близость случайных пассажей.
Здоровая модель ~0.3; коллапс ~>0.7. exit 2 если выше порога (чтобы пайплайн не
тратил 30 мин на full-eval вырожденной модели).

    python scripts/09_collapse_check.py --model runs/kaz-bge5 --thresh 0.6
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    c = load_dataset("issai/kazqad-retrieval", "corpus", split="train")
    step = max(1, len(c) // args.n)
    sub = c.select(list(range(0, len(c), step))[:args.n])
    passages = [((t or "") + ". " + (x or "")).strip() for t, x in zip(sub["title"], sub["text"])]

    m = SentenceTransformer(args.model, trust_remote_code=True,
                            model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 256
    V = m.encode(passages, batch_size=256, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    S = V @ V.T
    n = len(V)
    off = float((S.sum() - np.trace(S)) / (n * n - n))
    ok = off <= args.thresh
    print(f"MEANCOS={off:.3f} thresh={args.thresh} -> {'OK' if ok else 'COLLAPSE'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
