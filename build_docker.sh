#!/usr/bin/env bash
# Build linux/amd64 image (runs on Linux / Windows / macOS Docker hosts).
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-faceplugin/face-recognition-liveness-sdk:local}"

LIB_DRIVE="https://drive.google.com/drive/folders/1Lzz3eb_JMDZ0xyGtnGzxsUmMbgaYzin6?usp=sharing"

if [[ ! -f lib/cpu/libFaceRecognitionSDK.so ]] \
  || [[ ! -f lib/cpu/libfacerec.so ]] \
  || [[ ! -f lib/cpu/far.fpk ]] \
  || [[ ! -f lib/cpu/libfal-eng.so ]] \
  || [[ ! -f lib/cpu/fal.fpk ]]; then
  echo "ERROR: ./lib/cpu/ is incomplete."
  echo "Need: libFaceRecognitionSDK.so, libfacerec.so, far.fpk, libfal-eng.so, fal.fpk"
  echo "Download all files from Google Drive into ./lib/cpu/:"
  echo "  $LIB_DRIVE"
  exit 1
fi

docker build --platform linux/amd64 -t "$TAG" .
echo "Built $TAG (linux/amd64)"
