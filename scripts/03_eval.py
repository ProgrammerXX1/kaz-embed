#!/usr/bin/env python3
"""Единый харнесс оценки казахских эмбеддеров (честное сравнение с BGE-M3).
Ретрив-метрики считаются на GPU (torch.topk) — быстро даже на корпусе 25k+.

Задачи:
  * ktd    — KazakhTextDuplicates: content -> modified_content (тот же индекс). MRR/R@1/R@10.
  * infoqa — informatics_kaz: question -> context (корпус уник. контекстов). Acc@1/MRR.
  * kazqad — issai/kazqad-retrieval test: query -> positive (пул пассажей с hard-neg). nDCG@10/MRR/R@1.

    python scripts/03_eval.py --models models/bge-m3-st intfloat/multilingual-e5-large \
        sentence-transformers/LaBSE runs/kaz-bge/final runs/kaz-e5/final --tasks kazqad ktd infoqa
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
    m.eval()
    return m


def enc(model, texts, kind, e5):
    if e5:
        texts = [("query: " if kind == "query" else "passage: ") + t for t in texts]
    return model.encode(list(texts), batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def _metrics(Q, C, golds):
    """golds[i] = множество индексов правильных документов в C для запроса i.
    Топ-100 берём на GPU (torch.topk) — быстро."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Qt = torch.as_tensor(Q, device=dev, dtype=torch.float32)
    Ct = torch.as_tensor(C, device=dev, dtype=torch.float32)
    k = min(100, Ct.shape[0])
    n = len(golds)
    ndcg = mrr = r1 = r10 = 0.0
    B = 512
    for s in range(0, n, B):
        topi = (Qt[s:s + B] @ Ct.T).topk(k, dim=1).indices.cpu().numpy()
        for row, g in zip(topi, golds[s:s + B]):
            hits = [1 if j in g else 0 for j in row]
            dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits[:10]))
            ideal = sum(1 / np.log2(r + 2) for r in range(min(len(g), 10)))
            ndcg += dcg / ideal if ideal else 0.0
            mrr += next((1 / (r + 1) for r, h in enumerate(hits) if h), 0.0)
            r1 += 1.0 if hits[0] else 0.0
            r10 += 1.0 if any(hits[:10]) else 0.0
    return {"nDCG@10": ndcg / n, "MRR": mrr / n, "Recall@1": r1 / n,
            "Recall@10": r10 / n, "n": n}


def eval_ktd(model, e5):
    from datasets import load_dataset
    ds = load_dataset("Arailym-tleubayeva/KazakhTextDuplicates", split="train")
    Q = enc(model, ds["content"], "query", e5)
    C = enc(model, ds["modified_content"], "passage", e5)
    return _metrics(Q, C, [{i} for i in range(len(C))])


def eval_infoqa(model, e5):
    from datasets import load_dataset
    ds = load_dataset("Kundyzka/informatics_kaz", split="train")
    contexts = list(dict.fromkeys(ds["context"]))
    cidx = {c: i for i, c in enumerate(contexts)}
    Q = enc(model, ds["question"], "query", e5)
    C = enc(model, contexts, "passage", e5)
    m = _metrics(Q, C, [{cidx[c]} for c in ds["context"]])
    m["Acc@1"] = m["Recall@1"]; m["n_corpus"] = len(contexts)
    return m


def eval_kazqad(model, e5, split="test"):
    from datasets import load_dataset
    ds = load_dataset("issai/kazqad-retrieval", "queries-and-passages", split=split)
    doc = {}
    queries, golds = [], []
    for r in ds:
        pos = r.get("positive_passages") or []
        if not r["query"] or not pos:
            continue
        g = set()
        for p in pos:
            doc[p["docid"]] = p["text"]; g.add(p["docid"])
        for ngp in (r.get("negative_passages") or []):
            doc[ngp["docid"]] = ngp["text"]
        queries.append(r["query"]); golds.append(g)
    docids = list(doc)
    didx = {d: i for i, d in enumerate(docids)}
    golds = [{didx[d] for d in g} for g in golds]
    C = enc(model, [doc[d] for d in docids], "passage", e5)
    Q = enc(model, queries, "query", e5)
    m = _metrics(Q, C, golds)
    m["n_corpus"] = len(docids)
    return m


TASKS = {"ktd": eval_ktd, "infoqa": eval_infoqa, "kazqad": eval_kazqad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["kazqad", "ktd", "infoqa"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    lb_path = RESULTS / "leaderboard.json"
    table = json.loads(lb_path.read_text()) if lb_path.exists() else {}

    for name in args.models:
        e5 = needs_e5_prefix(name)
        print(f"\n=== {name} (e5_prefix={e5}) ===", flush=True)
        try:
            model = load_model(name)
        except Exception as ex:  # noqa: BLE001
            print(f"  LOAD FAILED: {str(ex)[:200]}")
            table[name] = {"load_error": str(ex)[:200]}
            lb_path.write_text(json.dumps(table, ensure_ascii=False, indent=2)); continue
        row = table.get(name, {})
        for t in args.tasks:
            try:
                row[t] = TASKS[t](model, e5)
            except Exception as ex:  # noqa: BLE001
                row[t] = {"error": str(ex)[:200]}
            print(f"  {t}: {json.dumps(row[t], ensure_ascii=False)}", flush=True)
        table[name] = row
        del model
        lb_path.write_text(json.dumps(table, ensure_ascii=False, indent=2))

    print(f"\nсводка -> {lb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
