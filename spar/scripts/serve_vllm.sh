#!/usr/bin/env bash
# vLLM OpenAI-compatible server
# 사용: ./scripts/serve_vllm.sh [--model <id>] [--port <n>] [--help]

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via env or CLI args)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-google/gemma-4-E4B-it}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"           # auto | float16 | bfloat16
QUANTIZATION="${QUANTIZATION:-}" # awq | gptq | squeezellm | "" (none)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

# ---------------------------------------------------------------------------
# CLI arg parse
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --model   <id>    HuggingFace model ID or local path (default: $MODEL)
  --port    <n>     Listening port (default: $PORT)
  --tp      <n>     Tensor parallel size (default: $TENSOR_PARALLEL_SIZE)
  --quant   <type>  Quantization: awq | gptq | (blank = none)
  --dtype   <type>  Data type: auto | float16 | bfloat16
  --gpu-mem <frac>  GPU memory utilization 0~1 (default: $GPU_MEMORY_UTILIZATION)
  --max-len <n>     Max model context length (default: $MAX_MODEL_LEN)
  -h, --help        Show this help

Env overrides (same names in upper case):
  MODEL, PORT, HOST, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN,
  TENSOR_PARALLEL_SIZE, DTYPE, QUANTIZATION, MAX_NUM_SEQS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2";                    shift 2 ;;
    --port)    PORT="$2";                     shift 2 ;;
    --tp)      TENSOR_PARALLEL_SIZE="$2";     shift 2 ;;
    --quant)   QUANTIZATION="$2";             shift 2 ;;
    --dtype)   DTYPE="$2";                    shift 2 ;;
    --gpu-mem) GPU_MEMORY_UTILIZATION="$2";   shift 2 ;;
    --max-len) MAX_MODEL_LEN="$2";            shift 2 ;;
    -h|--help) usage; exit 0                  ;;
    *) echo "Unknown arg: $1"; usage; exit 1  ;;
  esac
done

# ---------------------------------------------------------------------------
# Build vllm serve command
# ---------------------------------------------------------------------------
CMD=(
  vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --served-model-name "$(basename "$MODEL")"
  --trust-remote-code
)

[[ -n "$QUANTIZATION" ]] && CMD+=(--quantization "$QUANTIZATION")
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=(--chat-template "$CHAT_TEMPLATE")

# ---------------------------------------------------------------------------
# Print config summary
# ---------------------------------------------------------------------------
echo "============================================"
echo "  vLLM Server"
echo "============================================"
echo "  Model     : $MODEL"
echo "  Endpoint  : http://${HOST}:${PORT}/v1"
echo "  TP size   : $TENSOR_PARALLEL_SIZE"
echo "  GPU mem   : $GPU_MEMORY_UTILIZATION"
echo "  Max len   : $MAX_MODEL_LEN"
echo "  Dtype     : $DTYPE"
[[ -n "$QUANTIZATION" ]] && echo "  Quant     : $QUANTIZATION"
echo "============================================"
echo ""
echo "  Test:"
echo "    curl http://localhost:${PORT}/v1/models"
echo "    curl http://localhost:${PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"$(basename "$MODEL")\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
echo "============================================"
echo ""

exec "${CMD[@]}"
