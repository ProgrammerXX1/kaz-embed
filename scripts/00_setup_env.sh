#!/usr/bin/env bash
# Окружение для обучения казахской эмбеддинг-модели на debian3090 (RTX 3090, CUDA 12.x).
# Идемпотентно. Запуск на удалённой машине:
#   tailscale ssh root@debian3090 'bash -s' < scripts/00_setup_env.sh
set -euo pipefail

PROJ="${PROJ:-/root/kaz-embed}"
mkdir -p "$PROJ"/{data,models,runs,results}
cd "$PROJ"

# uv — быстрый менеджер пакетов/питона (py3.13 слишком свежий для части ML-стека → берём 3.11)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 "$PROJ/.venv"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"

# torch под CUDA 12.1 (драйвер 550 поддерживает)
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch

# основной стек
uv pip install \
  "sentence-transformers>=5" \
  "transformers>=4.44" "datasets" "accelerate" "peft" "bitsandbytes" \
  "FlagEmbedding" \
  "mteb==2.21.0" \
  "faiss-gpu-cu12" \
  "huggingface_hub[cli]" "hf_transfer" \
  "datasketch" "orjson" "pyyaml" "tqdm"

echo "HF_HUB_ENABLE_HF_TRANSFER=1" >> "$PROJ/.env"

python - <<'PY'
import torch, sentence_transformers as st
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("sentence-transformers", st.__version__)
PY

echo "OK. Активация: source $PROJ/.venv/bin/activate"
echo "Не забудь: huggingface-cli login   (для gated-датасетов)"
