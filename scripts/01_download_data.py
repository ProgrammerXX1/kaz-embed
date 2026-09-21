#!/usr/bin/env python3
"""Скачивание источников из data/datasets.yaml на debian3090.

По умолчанию — dry-run (только печатает план). Реальное скачивание: --run.
Можно фильтровать по назначению: --use pretrain,pairs,eval и пропускать
непроверенные:  --skip-verify.

Примеры:
    python scripts/01_download_data.py                      # план (dry-run)
    python scripts/01_download_data.py --run --use eval     # скачать только eval
    python scripts/01_download_data.py --run --skip-verify  # всё проверенное
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "datasets.yaml"
OUT = Path(os.environ.get("DATA_DIR", ROOT / "data" / "raw"))


def iter_entries(man: dict):
    for group, items in man.items():
        for it in items:
            if isinstance(it, dict):
                yield group, it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="реально скачивать (иначе dry-run)")
    ap.add_argument("--use", default="", help="фильтр по use через запятую")
    ap.add_argument("--skip-verify", action="store_true", help="пропустить verify:true")
    args = ap.parse_args()

    man = yaml.safe_load(MANIFEST.read_text())
    want = {u.strip() for u in args.use.split(",") if u.strip()}
    OUT.mkdir(parents=True, exist_ok=True)

    plan = []
    for group, it in iter_entries(man):
        repo = it.get("repo_id")
        if not repo:
            continue  # source-only (GitHub/OPUS) — качается отдельными рецептами
        if want and it.get("use") not in want:
            continue
        if args.skip_verify and it.get("verify"):
            continue
        plan.append((group, it))

    print(f"# {'СКАЧИВАНИЕ' if args.run else 'ПЛАН (dry-run)'} — {len(plan)} датасетов -> {OUT}\n")
    for group, it in plan:
        repo, sub = it["repo_id"], it.get("subset")
        gated = " [GATED: нужен токен]" if it.get("gated") else ""
        vfy = " [VERIFY]" if it.get("verify") else ""
        print(f"- {repo}" + (f" :: {sub}" if sub else "") + f"  ({it.get('use')}){gated}{vfy}")

    if not args.run:
        print("\nДобавь --run для скачивания. Gated-датасеты требуют `huggingface-cli login`.")
        return 0

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from datasets import load_dataset  # noqa: E402
    for group, it in plan:
        repo, sub = it["repo_id"], it.get("subset")
        name = repo.replace("/", "__") + (f"__{sub}" if sub else "")
        dst = OUT / group / name
        if dst.exists():
            print(f"[skip] уже есть: {dst}")
            continue
        print(f"[get ] {repo} {sub or ''} -> {dst}")
        try:
            ds = load_dataset(repo, sub) if sub else load_dataset(repo)
            Path(dst).mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(dst))
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {repo}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
