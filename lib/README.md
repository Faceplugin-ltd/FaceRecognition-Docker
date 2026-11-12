# Runtime libraries

Download the **FaceRecognition-Linux-fast** Drive folder into `cpu/`:

https://drive.google.com/drive/folders/1Lzz3eb_JMDZ0xyGtnGzxsUmMbgaYzin6?usp=sharing

Expected files (CPU):

- `libFaceRecognitionSDK.so` (built on Linux with `build_libraries/build.sh`)
- `libfacerec.so`, `libonnxruntime.so`, `libonnxruntime_providers_shared.so`
- `far.fpk`
- `libfal-eng.so`, `fal.fpk`

Then `./run.sh` or `docker compose up --build`.
