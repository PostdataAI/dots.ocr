#!/bin/bash
set -e  # Exit on error
set -o pipefail

# 1. Clone repo if not exists
if [ ! -d "DotsOCR" ]; then
    echo "[INFO] Cloning repository..."
    git clone https://github.com/rednote-hilab/dots.ocr.git
    mv dots.ocr DotsOCR
else
    echo "[INFO] Repository already exists."
fi

cd /workspace/DotsOCR

# 2. Install PyTorch + dependencies
echo "[INFO] Installing PyTorch..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install vLLM
echo "[INFO] Installing vLLM..."
pip install vllm==0.9.1 --extra-index-url https://download.pytorch.org/whl/cu128

# 4. Install DotsOCR in editable mode
echo "[INFO] Installing DotsOCR..."
pip install -e .

# 5. Download model weights
echo "[INFO] Downloading model weights..."
python tools/download_model.py

# 6. Set environment variables
export hf_model_path=./weights/DotsOCR
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH

# 7. Patch vLLM entrypoint
echo "[INFO] Patching vLLM entrypoint..."
sed -i '/^from vllm\.entrypoints\.cli\.main import main/a from DotsOCR import modeling_dots_ocr_vllm' $(which vllm)

# 8. Serve the model
echo "[INFO] Starting vLLM server..."
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --chat-template-content-format string \
    --served-model-name dotsocr \
    --api-key mission-mobile \
    --trust-remote-code