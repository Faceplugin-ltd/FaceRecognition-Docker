#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LIB_DRIVE="https://drive.google.com/drive/folders/1NVq0psW8PLfEX58FWNE-RKFWfCZdOMmz"

if [[ ! -f lib/cpu/libFaceRecognitionSDK.so ]] \
  || [[ ! -f lib/cpu/libfar-eng.so ]] \
  || [[ ! -f lib/cpu/far.fpk ]]; then
  echo "ERROR: ./lib/cpu/ is empty."
  echo "Download all files from Google Drive into ./lib/cpu/:"
  echo "  $LIB_DRIVE"
  exit 1
fi

export LICENSE="${LICENSE:-$(pwd)/license.txt}"
export LD_LIBRARY_PATH="$(pwd)/lib/cpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PORT="${PORT:-${FACESDK_PORT:-8083}}"
exec python3 app.py
