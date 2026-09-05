#!/usr/bin/env bash
# Build linux/amd64 image (runs on Linux / Windows / macOS Docker hosts).
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-faceplugin/face-recognition:local}"

LIB_DRIVE="https://drive.google.com/drive/folders/1NVq0psW8PLfEX58FWNE-RKFWfCZdOMmz"

if [[ ! -f lib/cpu/libFaceRecognitionSDK.so ]] \
  || [[ ! -f lib/cpu/libfar-eng.so ]] \
  || [[ ! -f lib/cpu/far.fpk ]]; then
  echo "ERROR: ./lib/cpu/ is incomplete (need libFaceRecognitionSDK.so, libfar-eng.so, far.fpk)."
  echo "Download all files from Google Drive into ./lib/cpu/:"
  echo "  $LIB_DRIVE"
  exit 1
fi

docker build --platform linux/amd64 -t "$TAG" .
echo "Built $TAG (linux/amd64)"
