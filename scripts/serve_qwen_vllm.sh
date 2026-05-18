#!/usr/bin/env bash
# Serve Qwen3-8B INT4 as local vLLM judge endpoint (target: RTX 4080 16GB).
set -euo pipefail

vllm serve Qwen/Qwen3-8B \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000
