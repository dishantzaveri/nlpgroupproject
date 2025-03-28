
# Long Context Reasoning with LLaMA on Multimodal Benchmarks

This repository provides a complete setup for evaluating LLaMA-based models (such as LLaMA-2-7B) on long-context multimodal tasks using **Full KV Cache (baseline)** on **MMMU** and **MILEBench** datasets.

## 📦 Contents

- `run_llama_kvcache.py` – Runs full KV cache inference using HuggingFace's LLaMA model
- `requirements.txt` – Required packages for the environment
- Instructions for downloading models, datasets, and running evaluation

---

## 🔧 Environment Setup

```bash
# Create and activate a conda environment
conda create -n llama-mmlm python=3.9 -y
conda activate llama-mmlm

# Install base packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets peft sentencepiece einops tqdm scikit-learn
