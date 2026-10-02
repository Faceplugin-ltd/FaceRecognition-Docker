<div align="center">
<img alt="FacePlugin" src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/brand/logo.png" width="400"/>
</div>

#### 🌐 Company Site - [Here](https://faceplugin.com)
#### 🤗 Hugging Face - [Here](https://huggingface.co/FacePlugin-Ltd)
#### 🛟 Help Center - [Here](https://doc.faceplugin.com)
#### 🐳 Docker Hub - [Here](https://hub.docker.com/r/faceplugin/face-recognition)

# FacePlugin Face Recognition SDK — Linux / Docker (Fully On-Premise)

> **Ready in minutes:** `docker pull` → copy machine code from logs → `curl /api/health`.  
> Jump: [Quick Start](#quick-start) · [Start the API](#start-the-api) · [SDK License](#sdk-license) · [Setup on your own app](#setup-on-your-own-app) · [Try it](#try-it)

## Quick Start

- [ ] Download and run the appropriate Docker image from [FacePlugin Docker Hub](https://hub.docker.com/r/faceplugin/face-recognition). [See Option A for details](#option-a--docker-hub-no-drive-download).
- [ ] **Confirm it is running:** `curl -s http://127.0.0.1:8083/api/health` (no license needed yet)
- [ ] [Contact us](#contact) with your machine code to obtain a license key, then activate with `POST /api/activate` — [SDK License](#sdk-license)
- [ ] **Try it:** Postman, curl, or local Gradio demo on **9003** (`demo.py`)

Docs: [https://doc.faceplugin.com](https://doc.faceplugin.com)

## Introduction

FacePlugin **Face Recognition SDK for Linux / Docker** is a fully on-premise biometric engine for KYC, access control, and identity verification. It runs face detection (bounding box, landmarks, pose, attributes), ICAO-style face quality, template extraction, 1:1 matching, and feature similarity — all on your server.

This repository is **standalone**. Pull from Docker Hub and run — **no other FacePlugin repository is required**.

All processing stays on your server. **No** biometric data is sent to FacePlugin cloud — built for banking, eKYC, and on-premise compliance workflows.

**One repository** for Linux SDK + Docker. Native libraries are **linux/amd64**; the Docker image runs on Linux, Windows, and macOS hosts via Docker (Apple Silicon uses amd64 emulation). This product is **CPU-only**.

Test with Postman, curl, or the local Gradio demo (`demo.py`) covering Detect, Quality, and Match. Docs: [https://doc.faceplugin.com](https://doc.faceplugin.com).

### Main Functionalities

| Feature | API |
| ------- | --- |
| Face detection (bounding box, landmarks, pose, attributes) | `POST /api/detect` · `sdk.detect` |
| Face quality analysis (ICAO-style checks) | `POST /api/quality` · `sdk.quality` |
| Face template extraction for matching | `POST /api/feature` · `sdk.feature` |
| 1:1 face match (two images) | `POST /api/match` · `sdk.match` |
| Feature vector similarity scoring | `POST /api/similarity` · `sdk.similarity` |
| Health / machine code / activate | `GET /api/health` · `GET /api/machinecode` · `POST /api/activate` |
| License capabilities | `GET /api/licenseStatus` · `sdk.get_license_status` |

### Product List

| Platform | Repository |
|----------|------------|
| Android (Recognition) | [FaceRecognition-Android](https://github.com/Faceplugin-ltd/FaceRecognition-Android) |
| iOS (Recognition) | [FaceRecognition-iOS](https://github.com/Faceplugin-ltd/FaceRecognition-iOS) |
| React Native (Recognition) | [FaceRecognition-React-Native](https://github.com/Faceplugin-ltd/FaceRecognition-React-Native) |
| Flutter (Recognition) | [FaceRecognition-Flutter](https://github.com/Faceplugin-ltd/FaceRecognition-Flutter) |
| Ionic Capacitor (Recognition) | [FaceRecognition-Ionic-Capacitor](https://github.com/Faceplugin-ltd/FaceRecognition-Ionic-Capacitor) |
| Ionic Cordova (Recognition) | [FaceRecognition-Ionic-Cordova](https://github.com/Faceplugin-ltd/FaceRecognition-Ionic-Cordova) |
| Windows (Recognition) | [FaceRecognition-Windows](https://github.com/Faceplugin-ltd/FaceRecognition-Windows) |
| **Linux / Docker (Recognition)** | **[FaceRecognition-Docker](https://github.com/Faceplugin-ltd/FaceRecognition-Docker)** (**this repo**) |
| Android (Liveness) | [FaceLivenessDetection-Android](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Android) |
| iOS (Liveness) | [FaceLivenessDetection-iOS](https://github.com/Faceplugin-ltd/FaceLivenessDetection-iOS) |
| Windows (Liveness) | [FaceLivenessDetection-Windows](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Windows) |
| Linux / Docker (Liveness) | [FaceLivenessDetection-Docker](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Docker) |

x
## Before you start

| Step | What you need |
| ---- | ------------- |
| 1 | A Linux host **or** Docker (Desktop or Engine) |
| 2 | Docker Hub pull does **not** need Drive — [see Option A](#option-a--docker-hub-no-drive-download) |
| 3 | Start **without** a license. Copy machine code from logs or `GET /api/machinecode`, send it to FacePlugin ([contact](#contact)), then activate with your license key |

You do **not** need a license to start the API once. Product endpoints unlock after you activate.

### System requirements

| Item | Minimum | Recommended |
| ---- | ------- | ----------- |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 4 GB | 8 GB |
| OS (Docker) | Linux + Docker Engine | Ubuntu 22.04 / 24.04 |

## Start the API

You can start **without** a license — the server prints your machine code on startup.

The API starts even if activation fails. Copy the **machine code** from the log and send it to FacePlugin.

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/unactivated.png" alt="Docker logs: machine code printed, activation failed, Flask API still listening" width="900"/>
</p>

### Option A — Docker Hub (no Drive download)

Runtime is already inside the image.

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

### Optional — Run multiple containers with one license

You only need this section if you want to run multiple Face Recognition containers on the same Linux host.

On Linux, mount `/etc/machine-id` into each container so they use the same machine code. Each container must have a different container name and host port.

For example:

```bash
sudo docker run -d --name faceplugin-face-recognition-2 \
  --shm-size=2gb --privileged \
  -p 8084:8083 \
  -v /etc/machine-id:/etc/machine-id:ro \
  faceplugin/face-recognition:latest
```

You can then activate each container using the same license key.

Note: On Docker Desktop (macOS/Windows), do not use the `/etc/machine-id` volume. Each container may require its own license.


### Need Docker Compose or a native install?

The steps above (Docker Hub) are enough for most teams. If you need **Docker Compose** with a local build, or a **native Linux** install without Docker Hub, [contact FacePlugin](#contact) and we will share the Drive runtime package and setup for your environment.


## SDK License

Licenses are **offline** and bound to your machine code. Offline cryptography is built into the SDK — no OpenSSL install.

### How to get a license

1. **Start the server** ([above](#start-the-api)) with Docker Hub. A license is not required for the first start.
2. **Copy the machine code** from container logs or `GET /api/machinecode`.
3. **Send that machine code** to FacePlugin ([contact](#contact)). We will issue a license key for that code.
4. **Activate** with the license key:

```bash
# Paste your license key into ./license.txt (overwrite the file).

# Detached Docker will not re-read license.txt on its own — POST the key:
curl -s -X POST http://127.0.0.1:8083/api/activate \
 -H 'Content-Type: text/plain' \
 --data-binary @license.txt

```

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/activate.png" alt="POST /api/activate with license.txt — success true" width="900"/>
</p>

Use the machine code from the environment you will run in production. **Docker and local host codes are different** — if you run in Docker, send the Docker machine code.

### License capabilities

After activation, `GET /api/licenseStatus` reports what the key unlocks. The Gradio demo shows the same summary as **License:** at the top of the page.

This App exposes **recognition** APIs only. Typical labels:

- **Recognition only** / **Recognition + Liveness** — Detect / Quality / Match
- **Liveness only** — recognition APIs stay unavailable on this App
- **Not licensed** — machine code only until you activate

```bash
curl -s http://127.0.0.1:8083/api/licenseStatus
```

## Try it

### Health

```bash
curl -s http://127.0.0.1:8083/api/health
```

### Documentation

[https://doc.faceplugin.com](https://doc.faceplugin.com)

### Postman

Import [`postman/FaceRecognition-API.postman_collection.json`](postman/FaceRecognition-API.postman_collection.json).

Default base URL: `http://127.0.0.1:8083`

Routes are `/api/*` (no version segment in paths).

### Demo UI (Gradio) — local only

The Docker image is **API/SDK server only** (no Gradio). For a local FacePlugin Face Recognition demo in the browser — Detect, Quality, and Match — on the host (API must already be running on port 8083). The header shows **License:** from `/api/licenseStatus`.

```bash
pip3 install -r requirements-demo.txt
DEMO_PORT=9003 API_BASE=http://127.0.0.1:8083 python3 demo.py
```

Open **[http://127.0.0.1:9003](http://127.0.0.1:9003)**. Examples when present: `assets/examples/samples/`.

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/demo-ui-detect.png" alt="FacePlugin Face Recognition Linux demo — Detect tab with landmarks and attributes" width="900"/>
</p>

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/demo-ui-quality.png" alt="FacePlugin Face Recognition Linux demo — Quality tab with ICAO-style checks" width="900"/>
</p>

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/desktop/demo-ui-match.png" alt="FacePlugin Face Recognition Linux demo — Match tab with 1:1 similarity scores" width="900"/>
</p>

Tabs: **Detect**, **Quality**, **Match**. Each action has a **Result** table (attributes, quality checks, or match scores) and **Raw JSON** for integration. Detect / Quality examples are every file under `assets/examples/samples/`. Match is **Odd vs Even**: pick one image from each group, then Match.

## Setup on your own app

Two ways to call the same engine. Full protocol: [https://doc.faceplugin.com](https://doc.faceplugin.com).

| Path | When to use |
| ---- | ----------- |
| **HTTP** (`app.py`) | Any language. Keep this API running and `POST` images as JSON. |
| **`sdk.py`** | Python on the **same** Linux host as `lib/cpu/` (or inside the container). No HTTP hop. |

**HTTP (any language):** start the API, then call `/api/detect`, `/api/quality`, `/api/match`, `/api/feature`, `/api/similarity`. Images are base64. See [Try it](#try-it) and Postman.

**Python in-process:** copy `sdk.py` + `lib/cpu/` into your project (or `import sdk` from this repo). Call order: `get_machine_code` → `activate` → `init_sdk` → detect / quality / feature / match / similarity. Check `get_license_status()` for recognition vs liveness flags. Return code `0` means success.

You do **not** need Gradio (`demo.py`) in production — it is a host-only test UI.

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

```python
status = sdk.get_license_status()
```

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

## Contact

<div align="left">
<a target="_blank" href="mailto:info@faceplugin.com"><img src="https://img.shields.io/badge/email-info@faceplugin.com-blue.svg?logo=gmail" alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://t.me/FacePluginSupport"><img src="https://img.shields.io/badge/telegram-@FacePluginSupport-blue.svg?logo=telegram" alt="Telegram @FacePluginSupport"></a>&emsp;
<a target="_blank" href="https://wa.me/+14692784822"><img src="https://img.shields.io/badge/whatsapp-faceplugin-blue.svg?logo=whatsapp" alt="faceplugin.com"></a>
</div>
