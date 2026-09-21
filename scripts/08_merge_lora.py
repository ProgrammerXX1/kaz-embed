#!/usr/bin/env python3
"""Сливает обученный LoRA-адаптер с базой bge-m3 -> полная модель (диагностика типов)."""
from __future__ import annotations
import argparse
import torch
from sentence_transformers import SentenceTransformer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/bge-m3-st")
    ap.add_argument("--adapter", default="runs/kaz-bge4/final")
    ap.add_argument("--out", default="runs/kaz-bge4-merged")
    args = ap.parse_args()

    m = SentenceTransformer(args.base, trust_remote_code=True,
                            model_kwargs={"torch_dtype": torch.float16})
    st_mod = m[0]
    am = st_mod.auto_model
    print("base auto_model type:", type(am).__name__)

    from peft import PeftModel
    peft_am = PeftModel.from_pretrained(am, args.adapter)
    print("after from_pretrained:", type(peft_am).__name__,
          "| has merge_and_unload:", hasattr(peft_am, "merge_and_unload"))

    merged = peft_am.merge_and_unload()
    print("merged type:", type(merged).__name__)
    st_mod.auto_model = merged
    print("reassigned auto_model type:", type(st_mod.auto_model).__name__)

    m.save_pretrained(args.out)
    v = m.encode(["Қазақстан Республикасының астанасы қай қала?",
                  "Астана — Қазақстанның астанасы.",
                  "Футбол ережелері туралы мақала."], normalize_embeddings=True)
    import numpy as np
    print(f"OK -> {args.out}  dim={v.shape[-1]}  "
          f"cos(q,pos)={float((v[0]*v[1]).sum()):.3f}  cos(q,neg)={float((v[0]*v[2]).sum()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
