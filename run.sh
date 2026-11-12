#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LIB_DRIVE="https://drive.google.com/drive/folders/1Lzz3eb_JMDZ0xyGtnGzxsUmMbgaYzin6?usp=sharing"

if [[ ! -f lib/cpu/libFaceRecognitionSDK.so ]] \
  || [[ ! -f lib/cpu/libfacerec.so ]] \
  || [[ ! -f lib/cpu/far.fpk ]] \
  || [[ ! -f lib/cpu/libfal-eng.so ]] \
  || [[ ! -f lib/cpu/fal.fpk ]]; then
  echo "ERROR: ./lib/cpu/ is incomplete."
  echo "Download all files from Google Drive into ./lib/cpu/:"
  echo "  $LIB_DRIVE"
  exit 1
fi

export LICENSE="${LICENSE:-$(pwd)/license.txt}"
export LD_LIBRARY_PATH="$(pwd)/lib/cpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Required for Face Liveness VFS interpositions (open/fopen/mmap hooks).
export LD_PRELOAD="$(pwd)/lib/cpu/libFaceRecognitionSDK.so${LD_PRELOAD:+:$LD_PRELOAD}"
export PORT="${PORT:-${FACESDK_PORT:-8083}}"
exec python3 app.py
