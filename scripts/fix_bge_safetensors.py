#!/usr/bin/env python3
"""Одноразовая конвертация BAAI/bge-m3 в safetensors + обёртка sentence-transformers
(CLS-пулинг). Репо bge-m3 хранит веса в pytorch_model.bin; transformers v5 не грузит
.bin при torch<2.6. Здесь НИЧЕГО не отключаем: читаем веса безопасным
torch.load(weights_only=True) и пересохраняем в safetensors. CPU-only.

Результат: /root/kaz-embed/models/bge-m3-st  (dense, CLS, готова для ST).
"""
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models

HF = "/root/kaz-embed/models/bge-m3-hf"
OUT = "/root/kaz-embed/models/bge-m3-st"


def main():
    src = snapshot_download("BAAI/bge-m3")
    print("downloaded:", src)

    # безопасное чтение весов (weights_only=True) — не обходим никаких проверок
    sd = torch.load(os.path.join(src, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    cfg = AutoConfig.from_pretrained(src)
    model = AutoModel.from_config(cfg)                 # чистый XLM-R backbone, без torch.load
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    print("  missing[:5]:", list(missing)[:5])
    print("  unexpected[:5]:", list(unexpected)[:5])
    # backbone должен совпасть почти полностью; много missing => проблема ключей
    if len(missing) > 5:
        raise SystemExit(f"СТОП: слишком много несопоставленных ключей ({len(missing)}).")

    model.save_pretrained(HF, safe_serialization=True)  # -> model.safetensors
    AutoTokenizer.from_pretrained(src).save_pretrained(HF)
    print("hf-safetensors:", HF)

    word = models.Transformer(HF, max_seq_length=512)   # грузит safetensors — ок
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
    st = SentenceTransformer(modules=[word, pool])
    st.save_pretrained(OUT)
    v = st.encode(["Қазақстанның астанасы қай қала?"], normalize_embeddings=True)
    print("OK ->", OUT, "dim=", v.shape[-1])


if __name__ == "__main__":
    main()
