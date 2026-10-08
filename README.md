<div align="center">
<img alt="FacePlugin" src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/brand/logo.png" width="400"/>
</div>

#### 🌐 Company Site - [Here](https://faceplugin.com)

#### 🤗 Hugging Face - [Here](https://huggingface.co/FacePlugin-Ltd)

#### 📚 Help Center - [Here](https://doc.faceplugin.com)

#### 🐳 Docker Hub - [Here](https://hub.docker.com/r/faceplugin/face-recognition)

# FacePlugin Face Recognition SDK — Linux / Docker (Fully On-Premise)

## Quick start

- **Docker (recommended):** `docker pull faceplugin/face-recognition:latest` then `docker run` — [Option A](#option-a--docker-hub)
- **Or local:** download CPU runtime into `lib/cpu/` — [Option B](#option-b--local-linux-runsh), then `./run.sh` — API on **8083**
- **Confirm it is running:** `curl -s http://127.0.0.1:8083/api/health` (no license needed yet)
- [Contact us](#contact) with your machine code to obtain a license key, then activate with `POST /api/activate` — [Activate your license](#activate-your-license)
- **Try it:** Postman, curl, or local Gradio demo on **9003** (`python3 demo.py`)

Docs: [doc.faceplugin.com](https://doc.faceplugin.com)\
Try online: [Hugging Face Space](https://huggingface.co/spaces/FacePlugin-Ltd/FaceRecognition-LivenessDetection-SDK)


## Introduction

**FacePlugin Face Recognition SDK** is a fully on-premise biometric engine for Linux and Docker. It runs face detection (bounding box, landmarks, pose, attributes), ICAO-style face quality analysis, template extraction, 1:1 matching, and feature similarity scoring.

The SDK supports **KYC, access control, and identity verification** workflows, and is built for banking, eKYC, and on-premise compliance. All processing runs on your own server, and **no images or biometric data are ever sent to FacePlugin**.

The SDK runs as a REST API server on Linux (x86_64), or through Docker on Linux, Windows, and macOS (Apple Silicon uses amd64 emulation). It runs on CPU only. This repository is self-contained, with no other FacePlugin repository required.

### Main Functionalities

| Feature                                                    | API                                                               |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| Face detection (bounding box, landmarks, pose, attributes) | `POST /api/detect` · `sdk.detect`                                 |
| Face quality analysis (ICAO-style checks)                  | `POST /api/quality` · `sdk.quality`                               |
| Face template extraction for matching                      | `POST /api/feature` · `sdk.feature`                               |
| 1:1 face match (two images)                                | `POST /api/match` · `sdk.match`                                   |
| Feature vector similarity scoring                          | `POST /api/similarity` · `sdk.similarity`                         |
| Health / machine code / activate                           | `GET /api/health` · `GET /api/machinecode` · `POST /api/activate` |
| License capabilities                                       | `GET /api/licenseStatus` · `sdk.get_license_status`               |

### Product List

| Platform                         | Repository                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Android (Recognition)            | [FaceRecognition-Android](https://github.com/Faceplugin-ltd/FaceRecognition-Android)                   |
| iOS (Recognition)                | [FaceRecognition-iOS](https://github.com/Faceplugin-ltd/FaceRecognition-iOS)                           |
| React Native (Recognition)       | [FaceRecognition-React-Native](https://github.com/Faceplugin-ltd/FaceRecognition-React-Native)         |
| Flutter (Recognition)            | [FaceRecognition-Flutter](https://github.com/Faceplugin-ltd/FaceRecognition-Flutter)                   |
| Ionic Capacitor (Recognition)    | [FaceRecognition-Ionic-Capacitor](https://github.com/Faceplugin-ltd/FaceRecognition-Ionic-Capacitor)   |
| Ionic Cordova (Recognition)      | [FaceRecognition-Ionic-Cordova](https://github.com/Faceplugin-ltd/FaceRecognition-Ionic-Cordova)       |
| Windows (Recognition)            | [FaceRecognition-Windows](https://github.com/Faceplugin-ltd/FaceRecognition-Windows)                   |
| **Linux / Docker (Recognition)** | **[FaceRecognition-Docker](https://github.com/Faceplugin-ltd/FaceRecognition-Docker)** (**this repo**) |
| Android (Liveness)               | [FaceLivenessDetection-Android](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Android)       |
| iOS (Liveness)                   | [FaceLivenessDetection-iOS](https://github.com/Faceplugin-ltd/FaceLivenessDetection-iOS)               |
| Windows (Liveness)               | [FaceLivenessDetection-Windows](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Windows)       |
| Linux / Docker (Liveness)        | [FaceLivenessDetection-Docker](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Docker)         |

---

## Start the API

You do **not** need a license to start the API. The server prints your machine code on startup, which you'll need to [activate your license](#activate-your-license). Product endpoints unlock after you activate.

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/unactivated.png" alt="Docker logs: machine code printed, activation failed, Flask API still listening" width="900"/>
</p>

### System requirements

| Item | Minimum | Recommended |
| ---- | ------- | ----------- |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 4 GB | 8 GB |
| OS (Docker) | Linux + Docker Engine | Ubuntu 22.04 / 24.04 |

### Option A — Docker Hub

The runtime is already inside the image, so no Google Drive download is needed.

```bash
sudo docker pull faceplugin/face-recognition:latest
sudo docker run -d --name faceplugin-face-recognition \
  --shm-size=2gb --privileged \
  -p 8083:8083 \
  -v /etc/machine-id:/etc/machine-id:ro \
  faceplugin/face-recognition:latest
sudo docker logs -f faceplugin-face-recognition
# Look for the machine code line in the logs
```

On Docker Desktop (macOS/Windows) omit the `/etc/machine-id` volume.

### Run multiple containers

To run multiple containers on one Linux host with a shared machine code / license, see the docs:

[https://doc.faceplugin.com/face-recognition-sdk/server-sdk/face-recognition-linux-sdk#run-multiple-containers](https://doc.faceplugin.com/face-recognition-sdk/server-sdk/face-recognition-linux-sdk#run-multiple-containers)

### Option B — Local Linux (`./run.sh`)

This option runs the server directly on your machine. It requires glibc **2.38 or newer** (for example, Ubuntu 24.04). Check your version with `ldd --version`. No GPU is needed; this product runs on CPU only.

#### 1. Clone the repository

```bash
git clone https://github.com/Faceplugin-ltd/FaceRecognition-Docker.git
cd FaceRecognition-Docker
```

#### 2. Download the runtime

The `lib/cpu/` folder is empty on GitHub because the native libraries and model files are too large to host there.

1. Open the [FaceRecognition Linux runtime folder on Google Drive](https://drive.google.com/drive/folders/1NVq0psW8PLfEX58FWNE-RKFWfCZdOMmz).
2. Download every file in the folder.
3. Place the files **directly** in `lib/cpu/`, not in a subfolder.

Your project should look like this:

```text
FaceRecognition-Docker/
└── lib/
    └── cpu/
        ├── libFaceRecognitionSDK.so
        ├── libfar-eng.so
        ├── far.fpk
        └── ... (remaining files from Google Drive)
```

> ⚠️ If Google Drive gives you a zip, extract it and move the files up so you don't end up with `lib/cpu/SomeFolder/libFaceRecognitionSDK.so`.

#### 3. Install dependencies and run

```bash
pip3 install -r requirements.txt
./run.sh
```

The API starts at **http://127.0.0.1:8083**, and the machine code is printed in the terminal. Continue with [Activate your license](#activate-your-license).

## Activate your license

Licenses work **offline** and are tied to the machine code of the environment where the server runs. Offline cryptography is built into the SDK, so no OpenSSL install is needed.

> ⚠️ **Docker and local installs have different machine codes.** Get the machine code from the same environment you'll use in production. If you'll run in Docker, send the code from the Docker container, not from the host.

1. **Start the server** using Docker Hub or `./run.sh` (see [Start the API](#start-the-api)). You don't need a license for the first start.
2. **Get your machine code.** It's printed in the startup log, or you can fetch it with `GET /api/machinecode`.
3. **Send the machine code to FacePlugin** ([contact us](#contact)). We'll reply with a license key for that machine code.
4. **Activate the license.** Save the license key to `license.txt` in the project root, replacing anything already in the file. Then send it to the running server:

   ```bash
   curl -s -X POST http://127.0.0.1:8083/api/activate \
     -H 'Content-Type: text/plain' \
     --data-binary @license.txt
   ```

   If you run in Docker, this command is required, because detached containers don't re-read `license.txt` after they start. The same command also works with `./run.sh`.

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/activate.png" alt="POST /api/activate with license.txt — success true" width="900"/>
</p>

### License capabilities

After activation, `GET /api/licenseStatus` reports what the key unlocks. The Gradio demo shows the same summary as **License:** at the top of the page.

This App exposes **recognition** APIs only. Typical labels:

- **Recognition only** / **Recognition + Liveness** — Detect / Quality / Match
- **Liveness only** — recognition APIs stay unavailable on this App
- **Not licensed** — machine code only until you activate

Check status anytime:

```bash
curl -s http://127.0.0.1:8083/api/licenseStatus
```

## Try it

### Health

```bash
curl -s http://127.0.0.1:8083/api/health
```

### Postman

Import [`postman/FaceRecognition-API.postman_collection.json`](postman/FaceRecognition-API.postman_collection.json).

Default base URL: `http://127.0.0.1:8083`


### Demo UI (Gradio) — local only

The Docker image includes only the API server, not the demo UI. To view results in your browser, run the Gradio demo on your own machine. Make sure the API is already running on port 8083 first.

```bash
pip3 install -r requirements-demo.txt
DEMO_PORT=9003 API_BASE=http://127.0.0.1:8083 python3 demo.py
```

Open **[http://127.0.0.1:9003](http://127.0.0.1:9003)** in your browser. Sample images, if included, are in `assets/examples/samples/`. The page header shows your current license status, taken from `/api/licenseStatus`.

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/demo-ui-detect.png" alt="FacePlugin Face Recognition Linux demo — Detect tab with landmarks and attributes" width="900"/>
</p>

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/demo-ui-quality.png" alt="FacePlugin Face Recognition Linux demo — Quality tab with ICAO-style checks" width="900"/>
</p>

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/demo-ui-match.png" alt="FacePlugin Face Recognition Linux demo — Match tab with 1:1 similarity scores" width="900"/>
</p>

- **Detect** — bounding box, landmarks, pose, and attributes
- **Quality** — ICAO-style face quality checks
- **Match** — 1:1 match scores; pick one image from the **Odd** group and one from the **Even** group, then Match

Each tab shows a **Result** table and **Raw JSON** for integration. Detect / Quality examples are every file under `assets/examples/samples/`.

---

## Setup on your own app

Two paths. You do **not** need the Gradio demo (`demo.py`) in production; it is a host-only test UI.

**HTTP** (any language) — start the API (see [Start the API](#start-the-api)) and keep it running, then `POST` base64 images as JSON to `/api/detect`, `/api/quality`, `/api/match`, `/api/feature`, or `/api/similarity`. See the Postman collection for request examples, and [doc.faceplugin.com](https://doc.faceplugin.com) for the full protocol.

**Python in-process** — on the **same** Linux host as `lib/cpu/` (or inside the container), copy [`sdk.py`](sdk.py) + `lib/cpu/` into your project (or `import sdk` from this repo) and call the SDK directly, with no HTTP hop. See [About SDK](#about-sdk).

---

## About SDK

Use the Python bindings in [`sdk.py`](sdk.py). Return code `0` means success.

### 1. Initializing the SDK

#### Step One

First, obtain the machine code for activation and request a license based on the machine code.

```python
import sdk

machine_code = sdk.get_machine_code()
print("machineCode:", machine_code) # machine code
```

#### Step Two

Next, activate the SDK with the path to your license file (`license.txt` containing your license key).

```python
ret = sdk.activate("license.txt")
```

If activation is successful, the return value will be `0`. Otherwise, an error value will be returned.

#### Step Three

After activation, call the initialization function of the SDK.

```python
ret = sdk.init_sdk()
```

If initialization is successful, the return value will be `0`. Otherwise, an error value will be returned.

### 2. APIs

#### Detect

```python
result = sdk.detect(base64_image, crop_image=False)
```

#### Quality

```python
result = sdk.quality(base64_image, crop_image=False)
```

#### Feature

```python
result = sdk.feature(base64_image)
```

#### Match

```python
result = sdk.match(base64_image1, base64_image2, crop_image=False)
```

#### Similarity

```python
result = sdk.similarity(feature1_b64, feature2_b64)
```

#### License status

```python
status = sdk.get_license_status()
```

HTTP endpoints: `/api/health`, `/api/machinecode`, `/api/licenseStatus`, `/api/backend`, `/api/activate`, `/api/detect`, `/api/quality`, `/api/match`, `/api/feature`, `/api/similarity`.


## Contact

Request a license, machine-code activation (machine code → license key), or integration help:

<div align="left">
<a target="_blank" href="mailto:info@faceplugin.com"><img src="https://img.shields.io/badge/email-info@faceplugin.com-blue.svg?logo=gmail" alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://t.me/FacePluginSupport"><img src="https://img.shields.io/badge/telegram-@FacePluginSupport-blue.svg?logo=telegram" alt="Telegram @FacePluginSupport"></a>&emsp;
<a target="_blank" href="https://wa.me/+14692784822"><img src="https://img.shields.io/badge/whatsapp-faceplugin-blue.svg?logo=whatsapp" alt="faceplugin.com"></a>
</div>
