#!/usr/bin/env python3
"""Честная оценка на ПОЛНОМ корпусе KazQAD (825k пассажей) — настоящий retrieval-бенч.
Каждый test-запрос ранжируется против всех 825k; gold = positive docids.
Метрики: nDCG@10, MRR, Recall@1/10/100. GPU-topk (fp16).

    python scripts/07_eval_fullcorpus.py --models models/bge-m3-st runs/kaz-bge2/final \
        intfloat/multilingual-e5-large
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def needs_e5_prefix(name: str) -> bool:
    import re
    # "e5" как отдельный токен (не часть "bge5"): перед ним не буква/цифра
    return bool(re.search(r"(?<![a-z0-9])e5", name.lower()))


def load_model(name: str):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, trust_remote_code=True,
                            model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 256          # пассажи короткие; иначе bge-m3 берёт дефолт 8192 и тормозит
    m.eval()
    return m


def enc(model, texts, kind, e5, bs=512):
    if e5:
        texts = [("query: " if kind == "query" else "passage: ") + t for t in texts]
    return model.encode(list(texts), batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float16)


def metrics_fullcorpus(Q, C, golds):
    """golds[i] = set индексов правильных пассажей в C. topk по всему корпусу на GPU."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Ct = torch.as_tensor(C, device=dev)                 # (825k, dim) fp16
    k = 100
    n = len(golds)
    ndcg = mrr = r1 = r10 = r100 = 0.0
    B = 64
    for s in range(0, n, B):
        Qt = torch.as_tensor(Q[s:s + B], device=dev)
        topi = (Qt @ Ct.T).topk(k, dim=1).indices.cpu().numpy()
        for row, g in zip(topi, golds[s:s + B]):
            hits = [1 if j in g else 0 for j in row]
            dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits[:10]))
            ideal = sum(1 / np.log2(r + 2) for r in range(min(len(g), 10)))
            ndcg += dcg / ideal if ideal else 0.0
            mrr += next((1 / (r + 1) for r, h in enumerate(hits) if h), 0.0)
            r1 += 1.0 if hits[0] else 0.0
            r10 += 1.0 if any(hits[:10]) else 0.0
            r100 += 1.0 if any(hits) else 0.0
    return {"nDCG@10": ndcg / n, "MRR": mrr / n, "Recall@1": r1 / n,
            "Recall@10": r10 / n, "Recall@100": r100 / n, "n": n, "n_corpus": len(C)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit-corpus", type=int, default=0, help=">0 — обрезать корпус (screening)")
    ap.add_argument("--out", default="leaderboard_fullcorpus.json", help="имя файла в results/")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    from datasets import load_dataset
    corpus = load_dataset("issai/kazqad-retrieval", "corpus", split="train")
    if args.limit_corpus:
        corpus = corpus.select(range(args.limit_corpus))
    docids = corpus["docid"]
    didx = {d: i for i, d in enumerate(docids)}
    titles, texts = corpus["title"], corpus["text"]
    passages = [((t or "") + ". " + (x or "")).strip() for t, x in zip(titles, texts)]
    print(f"корпус: {len(passages)} пассажей")

    qds = load_dataset("issai/kazqad-retrieval", "queries-and-passages", split="test")
    queries, golds = [], []
    dropped = 0
    for r in qds:
        pos = r.get("positive_passages") or []
        g = {didx[p["docid"]] for p in pos if p["docid"] in didx}
        if r["query"] and g:
            queries.append(r["query"]); golds.append(g)
        else:
            dropped += 1
    print(f"запросов: {len(queries)} (пропущено без gold в корпусе: {dropped})")

    lb = RESULTS / args.out
    table = json.loads(lb.read_text()) if lb.exists() else {}
    for name in args.models:
        e5 = needs_e5_prefix(name)
        print(f"\n=== {name} (e5={e5}) ===", flush=True)
        model = load_model(name)
        C = enc(model, passages, "passage", e5)
        Q = enc(model, queries, "query", e5)
        res = metrics_fullcorpus(Q, C, golds)
        table[name] = res
        print(f"  {json.dumps(res, ensure_ascii=False)}", flush=True)
        lb.write_text(json.dumps(table, ensure_ascii=False, indent=2))
        del model, C, Q
        import torch, gc; gc.collect(); torch.cuda.empty_cache()

    print(f"\n-> {lb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
