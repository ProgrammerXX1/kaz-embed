#!/usr/bin/env python3
"""Контрастивное дообучение казахского эмбеддера (sentence-transformers v5).

Под RTX 3090 (24 ГБ): bf16 + gradient checkpointing + CachedMultipleNegativesRankingLoss
(GradCache — большой эффективный батч при малой памяти) в обёртке MatryoshkaLoss.

Обучает ОДНУ базу за запуск (по решению — гоняем обе: gte и bge-m3):
    python scripts/04_train.py --base gte     # Alibaba-NLP/gte-multilingual-base
    python scripts/04_train.py --base bge      # BAAI/bge-m3

Вход: JSONL с полями {"anchor","positive","negative"} (триплеты после hard-neg mining),
либо {"anchor","positive"} (in-batch negatives). Путь: --train data/pairs/train.jsonl
"""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASES = {
    "gte": "Alibaba-NLP/gte-multilingual-base",
    "bge": "BAAI/bge-m3",
    "e5": "intfloat/multilingual-e5-large",
    "arctic": "Snowflake/snowflake-arctic-embed-l-v2.0",
}
DIMS = {"gte": [768, 512, 256, 128, 64]}          # остальные 1024-мерные
DIMS_DEFAULT = [1024, 768, 512, 256, 128]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=list(BASES), default="gte")
    ap.add_argument("--train", default=str(ROOT / "data/pairs/train.jsonl"))
    ap.add_argument("--eval", default=str(ROOT / "data/pairs/dev.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--mini-batch", type=int, default=16, help="микробатч GradCache")
    ap.add_argument("--batch", type=int, default=256, help="батч на шаг (эфф. батч растёт in-batch)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 — preflight/отладка")
    ap.add_argument("--grad-ckpt", choices=["auto", "on", "off"], default="auto",
                    help="auto: off для gte (кастомный modeling ломается), on для остальных")
    ap.add_argument("--loss", choices=["mnrl", "cached", "marginmse", "gist"], default="mnrl",
                    help="mnrl; cached=GradCache; marginmse=дистилляция; gist=CachedGIST (guide маскирует ложные негативы)")
    ap.add_argument("--temp", type=float, default=0.02, help="температура для gist/cached")
    ap.add_argument("--guide", default="models/bge-m3-st", help="guide-модель для GIST")
    ap.add_argument("--matryoshka", choices=["on", "off"], default="off")
    ap.add_argument("--init", default="", help="путь/id стартовой модели (override базы, напр. runs/kaz-bge2/final)")
    ap.add_argument("--lora", type=int, default=0, help=">0 — rank LoRA (не портит базу); 0 = full FT")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 => alpha=rank (scaling 1.0)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import (
        CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss,
        MatryoshkaLoss, MarginMSELoss,
    )
    from sentence_transformers.training_args import BatchSamplers

    if args.init:                                # старт с готового чекпоинта (напр. kaz-bge2)
        model_name = args.init
    elif args.base == "bge":                     # bge-m3 хранится в .bin → safetensors-конверт
        local = ROOT / "models" / "bge-m3-st"
        model_name = str(local) if local.exists() else BASES[args.base]
    else:
        model_name = BASES[args.base]
    out = args.out or str(ROOT / f"runs/kaz-{args.base}")
    print(f"base={model_name}  out={out}")

    model = SentenceTransformer(
        model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        trust_remote_code=True,   # gte-multilingual требует
    )
    model.max_seq_length = args.max_seq

    grad_ckpt = {"on": True, "off": False}.get(args.grad_ckpt, args.base != "gte")
    if grad_ckpt and hasattr(model[0].auto_model, "gradient_checkpointing_enable"):
        model[0].auto_model.gradient_checkpointing_enable()
    print(f"grad_checkpointing={grad_ckpt}")

    if args.lora > 0:                            # LoRA: не стираем базу, добавляем казахский
        from peft import LoraConfig
        alpha = args.lora_alpha if args.lora_alpha > 0 else args.lora  # scaling=alpha/rank
        model.add_adapter(LoraConfig(
            r=args.lora, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"],
            task_type="FEATURE_EXTRACTION"))
        print(f"LoRA r={args.lora} alpha={alpha}")

    def prep(dset):
        cols = [c for c in dset.column_names if c != "src"]   # anchor, positive, [negative*...]
        dset = dset.select_columns(cols)
        if args.base == "e5":   # e5 требует префиксы: query: к anchor, passage: к остальным
            def _pfx(b):
                out = {"anchor": ["query: " + t for t in b["anchor"]]}
                for c in cols:
                    if c not in ("anchor", "label"):
                        out[c] = ["passage: " + t for t in b[c]]
                return out
            dset = dset.map(_pfx, batched=True)
        return dset

    ds = prep(load_dataset("json", data_files={"train": args.train}, split="train"))
    eval_ds = None
    if Path(args.eval).exists():
        eval_ds = prep(load_dataset("json", data_files={"eval": args.eval}, split="eval"))

    if args.loss == "marginmse":
        loss = MarginMSELoss(model)                  # дистилляция: student margin ~ teacher margin
    elif args.loss == "gist":
        from sentence_transformers.losses import CachedGISTEmbedLoss
        guide = SentenceTransformer(args.guide, trust_remote_code=True,
                                    model_kwargs={"torch_dtype": torch.float16})
        loss = CachedGISTEmbedLoss(model, guide, mini_batch_size=args.mini_batch,
                                   temperature=args.temp)
    elif args.loss == "cached":
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    else:
        loss = MultipleNegativesRankingLoss(model)   # быстрый in-batch (63 негатива при batch 64)
    if args.matryoshka == "on":
        loss = MatryoshkaLoss(model, loss, matryoshka_dims=DIMS.get(args.base, DIMS_DEFAULT))
    print(f"loss={args.loss} matryoshka={args.matryoshka}")

    targs = SentenceTransformerTrainingArguments(
        output_dir=out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        bf16=True,
        optim="adamw_bnb_8bit",          # 8-bit Adam — экономит ~3 ГБ
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        dataloader_num_workers=4,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=ds, eval_dataset=eval_ds, loss=loss,
    )
    trainer.train()
    if args.lora > 0:                            # вплавляем LoRA в базу → обычная модель для eval
        try:
            model[0].auto_model = model[0].auto_model.merge_and_unload()
            print("LoRA merged into base")
        except Exception as e:  # noqa: BLE001
            print("merge warn:", str(e)[:120])
    model.save_pretrained(f"{out}/final")
    print(f"saved -> {out}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
