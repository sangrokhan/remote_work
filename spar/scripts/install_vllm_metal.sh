#!/usr/bin/env bash
# macOS Apple Silicon 전용 vllm-metal 설치 래퍼

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is only supported on macOS." >&2
  exit 1
fi

if [[ "${UV:-}" == "" ]]; then
  echo "UV path is not set." >&2
  exit 1
fi

if [[ ! -x "${UV}" ]]; then
  echo "uv executable not found at: ${UV}" >&2
  exit 1
fi

VENV_DIR="${VENV:-.venv}"
PROJECT_ROOT="$(pwd)"
INSTALL_ROOT="${PROJECT_ROOT}/${VENV_DIR}"
UV_BIN="$(cd "$(dirname "${UV}")" && pwd)/$(basename "${UV}")"
VLLM_METAL_VERSION="${VLLM_METAL_VERSION:-0.20.0}"
WORKDIR="$(mktemp -d)"
VLLM_TARBALL="vllm-${VLLM_METAL_VERSION}.tar.gz"
VLLM_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_METAL_VERSION}/${VLLM_TARBALL}"
PLUGIN_WHEEL_URL="${VLLM_METAL_WHEEL_URL:-}"

cleanup() {
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

echo "============================================"
echo "  vllm-metal Installer"
echo "============================================"
echo "  Install root : ${INSTALL_ROOT}"
echo "  vLLM version : ${VLLM_METAL_VERSION}"
echo "============================================"

if [[ -z "${PLUGIN_WHEEL_URL}" ]]; then
  PLUGIN_WHEEL_URL="$(
    curl -fsSL "https://api.github.com/repos/vllm-project/vllm-metal/releases/latest" \
      | /usr/bin/python3 -c '
import json
import sys

data = json.load(sys.stdin)
for asset in data.get("assets", []):
    url = asset.get("browser_download_url", "")
    if url.endswith(".whl"):
        print(url)
        break
'
  )"
fi

if [[ -z "${PLUGIN_WHEEL_URL}" ]]; then
  echo "Failed to resolve the latest vllm-metal wheel." >&2
  exit 1
fi

cd "${WORKDIR}"
curl -fsSL -O "${VLLM_URL}"
tar xf "${VLLM_TARBALL}"
cd "vllm-${VLLM_METAL_VERSION}"

"${UV_BIN}" pip install -r requirements/cpu.txt --python "${INSTALL_ROOT}/bin/python" --index-strategy unsafe-best-match

# Apple Clang 21+에서 chained comparison 경고가 에러로 승격되는 문제를 우회한다.
CXXFLAGS="-Wno-parentheses" "${UV_BIN}" pip install . --python "${INSTALL_ROOT}/bin/python"
"${UV_BIN}" pip install "${PLUGIN_WHEEL_URL}" --python "${INSTALL_ROOT}/bin/python"

echo ""
echo "vllm-metal installation complete."
