#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Inside Docker, a plain `docker run` has the container's own /etc/machine-id.
# The license is then bound to that id, not the host, and will not match later.
require_docker_launch() {
  [[ -f /.dockerenv ]] || return 0
  local ok=1
  local cap shm_kb
  if ! mountpoint -q /etc/machine-id; then
    ok=0
  fi
  shm_kb="$(df -k /dev/shm | awk 'NR==2 {print $2}')"
  if [[ "${shm_kb:-0}" -lt 1900000 ]]; then
    ok=0
  fi
  # Privileged sets CAP_SYS_ADMIN. A normal docker run does not.
  if ! python3 -c 'import pathlib; c=int(pathlib.Path("/proc/1/status").read_text().split("CapEff:")[1].split()[0], 16); raise SystemExit(0 if c & (1 << 21) else 1)'; then
    ok=0
  fi
  if [[ "$ok" -eq 1 ]]; then
    return 0
  fi
  echo "ERROR: Do not start this image with a plain docker run." >&2
  echo "That uses the container's own machine id, so the license machine code is wrong." >&2
  echo >&2
  echo "Start it with the host machine id, 2GB shared memory, and privileged mode:" >&2
  echo >&2
  echo "docker run -d --name faceplugin-face-recognition \\" >&2
  echo "  --shm-size=2gb --privileged \\" >&2
  echo "  -p 8092:8092 \\" >&2
  echo "  -v /etc/machine-id:/etc/machine-id:ro \\" >&2
  echo "  faceplugin/face-recognition:latest" >&2
  exit 1
}

require_docker_launch

LIB_DRIVE="https://drive.google.com/drive/folders/17TTpOkkc2ECrOsYP04RZnnUuIPhAhwJo?usp=sharing"

# Either SDK may fill lib/cpu/. Standard ships libonnxruntime.so as a real file.
# Plus ships that name as a link to libonnxruntime.so.1. -f follows the link, so a
# dangling Plus link left behind by a Standard export still fails here.
onnx_ok=0
if [[ -f lib/cpu/libonnxruntime.so || -f lib/cpu/libonnxruntime.so.1 ]]; then
  onnx_ok=1
fi
if [[ ! -f lib/cpu/libFaceRecognitionSDK.so ]] \
  || [[ ! -f lib/cpu/libfacerec.so ]] \
  || [[ ! -f lib/cpu/far.fpk ]] \
  || [[ "$onnx_ok" -ne 1 ]] \
  || [[ ! -f lib/cpu/libonnxruntime_providers_shared.so ]]; then
  echo "ERROR: ./lib/cpu/ is incomplete."
  echo "Need libFaceRecognitionSDK.so, libfacerec.so, far.fpk, libonnxruntime.so"
  echo "(or libonnxruntime.so.1), and libonnxruntime_providers_shared.so."
  echo "Download all files from Google Drive into ./lib/cpu/:"
  echo "  $LIB_DRIVE"
  exit 1
fi
if [[ ! -e lib/cpu/libonnxruntime.so && -f lib/cpu/libonnxruntime.so.1 ]]; then
  ln -s libonnxruntime.so.1 lib/cpu/libonnxruntime.so
fi

export LICENSE="${LICENSE:-$(pwd)/license.txt}"
export LD_LIBRARY_PATH="$(pwd)/lib/cpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PORT="${PORT:-${FACESDK_PORT:-8092}}"
exec python3 app.py
