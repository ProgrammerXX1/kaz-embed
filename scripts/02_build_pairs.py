#!/usr/bin/env python3
"""Сборка обучающих пар (anchor, positive) под реальные схемы датасетов.
Выход: data/pairs/train.jsonl, dev.jsonl.

Источники (Research-only — можно всё):
  * Zerde-QA-50K (kurumikz/Zerde-QA-50K): question -> answer  (нативный kk QA)
  * KazParC (issai/kazparc, config 'kazparc'): kk-сторона -> перевод (kk_en/kk_ru)
      1.74M пар; кросс-язычный якорь (регуляризует, поднимает retrieval)
  * (опц.) NLLB en-kk потоково + LaBSE-фильтр
  * (позже) синтетика из 02b (перевод mMARCO/NLI через tilmash + генерация KazLLM)

    python scripts/02_build_pairs.py --sources zerde kazparc \
        --kazparc-pairs kk_en kk_ru --kazparc-limit 400000
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pairs"
random.seed(13)


def from_zerde():
    from datasets import load_dataset
    ds = load_dataset("kurumikz/Zerde-QA-50K", split="train")
    out = [{"anchor": r["question"], "positive": r["answer"], "src": "zerde"}
           for r in ds if r["question"] and r["answer"]]
    print(f"  [zerde] {len(out)} пар")
    return out


def from_kazparc(want_pairs, limit):
    from datasets import load_dataset
    ds = load_dataset("issai/kazparc", "kazparc", split="train")
    out = []
    for r in ds:
        pair = r["pair"]                      # e.g. kk_en, en_kk, kk_ru, ru_kk
        if "kk" not in pair:
            continue
        norm = "kk_" + pair.replace("kk", "").replace("_", "")
        if want_pairs and norm not in want_pairs:
            continue
        # kk-сторона = anchor
        kk = r["source_lang"] if pair.startswith("kk") else r["target_lang"]
        oth = r["target_lang"] if pair.startswith("kk") else r["source_lang"]
        if kk and oth:
            out.append({"anchor": kk, "positive": oth, "src": f"kazparc:{norm}"})
        if limit and len(out) >= limit:
            break
    print(f"  [kazparc] {len(out)} пар ({want_pairs or 'все kk-*'})")
    return out


def from_kazqad_nq(limit):
    """issai/kazqad nq-translate-kk: question -> context (61k, kk QA)."""
    from datasets import load_dataset
    ds = load_dataset("issai/kazqad", "nq-translate-kk", split="train")
    out = [{"anchor": r["question"], "positive": r["context"], "src": "kazqad_nq"}
           for r in ds if r["question"] and r["context"]]
    if limit:
        out = out[:limit]
    print(f"  [kazqad_nq] {len(out)} пар")
    return out


def from_kazqad_rp():
    """issai/kazqad-retrieval queries-and-passages TRAIN ONLY: query -> positive.
    (test/validation НЕ трогаем — они для оценки; иначе утечка.)"""
    from datasets import load_dataset
    ds = load_dataset("issai/kazqad-retrieval", "queries-and-passages", split="train")
    out = []
    for r in ds:
        pos = r.get("positive_passages") or []
        if r["query"] and pos and pos[0].get("text"):
            out.append({"anchor": r["query"], "positive": pos[0]["text"], "src": "kazqad_rp"})
    print(f"  [kazqad_rp] {len(out)} пар (train-сплит)")
    return out


def from_nllb(limit, labse_thresh):
    from datasets import load_dataset
    out = []
    try:
        ds = load_dataset("allenai/nllb", "eng_Latn-kaz_Cyrl", split="train", streaming=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [nllb] verify: {e}"); return out
    for i, r in enumerate(ds):
        if i >= limit:
            break
        t = r.get("translation", r)
        en, kk = t.get("eng_Latn"), t.get("kaz_Cyrl")
        if en and kk:
            out.append({"anchor": kk, "positive": en, "src": "nllb"})
    print(f"  [nllb] {len(out)} сырых пар")
    if labse_thresh > 0:
        out = labse_filter(out, labse_thresh)
    return out


def labse_filter(pairs, thresh):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/LaBSE",
                            model_kwargs={"torch_dtype": torch.float16})
    A = m.encode([p["anchor"] for p in pairs], batch_size=128, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=True)
    B = m.encode([p["positive"] for p in pairs], batch_size=128, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=True)
    keep = [p for p, a, b in zip(pairs, A, B) if float((a * b).sum()) > thresh]
    print(f"  LaBSE: {len(keep)}/{len(pairs)} (>{thresh})")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["zerde", "kazqad_nq", "kazqad_rp", "kazparc"])
    ap.add_argument("--kazparc-pairs", nargs="+", default=["kk_en", "kk_ru"])
    ap.add_argument("--kazparc-limit", type=int, default=300000)
    ap.add_argument("--kazqad-nq-limit", type=int, default=0)
    ap.add_argument("--nllb-limit", type=int, default=0)
    ap.add_argument("--labse-thresh", type=float, default=0.0)
    ap.add_argument("--dev-frac", type=float, default=0.01)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = []
    if "zerde" in args.sources:
        pairs += from_zerde()
    if "kazqad_nq" in args.sources:
        pairs += from_kazqad_nq(args.kazqad_nq_limit)
    if "kazqad_rp" in args.sources:
        pairs += from_kazqad_rp()
    if "kazparc" in args.sources:
        pairs += from_kazparc(set(args.kazparc_pairs), args.kazparc_limit)
    if "nllb" in args.sources:
        pairs += from_nllb(args.nllb_limit, args.labse_thresh)

    random.shuffle(pairs)
    n_dev = max(1, int(len(pairs) * args.dev_frac))
    dev, train = pairs[:n_dev], pairs[n_dev:]
    with (OUT / "train.jsonl").open("w") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (OUT / "dev.jsonl").open("w") as f:
        for p in dev:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\ntrain={len(train)}  dev={len(dev)}  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
