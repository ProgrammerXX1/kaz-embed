#!/usr/bin/env python3
"""WiSE-FT: интерполяция весов θ = θ_base + α·(θ_ft − θ_base).
Отменяет забывание без обучения. α<1 тянет к базе (сохраняет силу bge-m3),
оставляя часть казахской специализации. Оба чекпоинта — полные (merged) модели.

    python scripts/10_wise_ft.py --base models/bge-m3-st --ft runs/kaz-bge5 \
        --alpha 0.4 --out runs/wise-0.4
"""
from __future__ import annotations
import argparse
import torch
from sentence_transformers import SentenceTransformer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/bge-m3-st")
    ap.add_argument("--ft", required=True, help="полная (merged) дообученная модель")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = SentenceTransformer(args.base, trust_remote_code=True,
                               model_kwargs={"torch_dtype": torch.float32})
    ft = SentenceTransformer(args.ft, trust_remote_code=True,
                             model_kwargs={"torch_dtype": torch.float32})
    bsd = base[0].auto_model.state_dict()
    fsd = ft[0].auto_model.state_dict()

    a = args.alpha
    new = {}
    for k, bt in bsd.items():
        ftk = fsd.get(k)
        if ftk is not None and bt.is_floating_point() and ftk.shape == bt.shape:
            new[k] = bt + a * (ftk.to(bt.dtype) - bt)     # WiSE-FT
        else:
            new[k] = bt                                    # буферы/несовпадения — из базы
    base[0].auto_model.load_state_dict(new)
    base.save_pretrained(args.out)

    v = base.encode(["Қазақстан Республикасының астанасы қай қала?",
                     "Астана — Қазақстанның астанасы.",
                     "Футбол ережелері туралы мақала."], normalize_embeddings=True)
    print(f"OK a={a} -> {args.out}  cos(q,pos)={float((v[0]*v[1]).sum()):.3f} "
          f"cos(q,neg)={float((v[0]*v[2]).sum()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
