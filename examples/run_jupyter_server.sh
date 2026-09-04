#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Start JupyterLab for amd_flashinfer_rocm_tutorial.ipynb (or any notebook in the repo).
#
# Usage (activate your ROCm/flashinfer env first, e.g. inside a container built
# from docker/Dockerfile.rocm):
#   cd /path/to/flashinfer
#   ./examples/run_jupyter_server.sh
#
# By default the server listens on 127.0.0.1 (localhost only).
# Remote / SSH port-forwarding: forward the port from your local machine:
#   ssh -L 8888:localhost:8888 user@node
# Then open the printed http://127.0.0.1:8888/lab?token=... in a browser.
#
# Override port:  JUPYTER_PORT=8890 ./examples/run_jupyter_server.sh
# Override IP:    JUPYTER_IP=0.0.0.0 ./examples/run_jupyter_server.sh
#   (setting JUPYTER_IP=0.0.0.0 binds on all interfaces; only do this intentionally)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! python -c "import jupyterlab" 2>/dev/null; then
  echo "Installing jupyterlab into the current Python environment..."
  python -m pip install jupyterlab
fi

PORT="${JUPYTER_PORT:-8888}"
IP="${JUPYTER_IP:-127.0.0.1}"

if [[ "$IP" == "0.0.0.0" ]]; then
  echo "WARNING: JUPYTER_IP=0.0.0.0 binds JupyterLab on ALL network interfaces."
  echo "         This exposes the server beyond localhost, which is risky on shared"
  echo "         machines or when --network=host is in use."
  echo "         Prefer the default 127.0.0.1 and use SSH -L for remote access."
  echo ""
fi

echo "Starting JupyterLab from: $ROOT"
echo "  URL: http://127.0.0.1:${PORT}/lab (use SSH -L if remote)"
echo "  Stop: Ctrl+C"
echo ""

exec python -m jupyter lab \
  --no-browser \
  --ip="$IP" \
  --port="$PORT" \
  --notebook-dir="$ROOT" \
  "$@"
