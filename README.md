<div align="center">
<img alt="FacePlugin" src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/brand/logo.png" width="400"/>
</div>

#### 🌐 Company Site - [Here](https://faceplugin.com)
#### 🤗 Hugging Face - [Here](https://huggingface.co/FacePlugin-Ltd)
#### 🛟 Help Center - [Here](https://doc.faceplugin.com)
#### 🐳 Docker Hub - [Here](https://hub.docker.com/r/faceplugin/face-recognition)

# FacePlugin Face Recognition + 1:N Identification — Docker (Fully On-Premise)

Jump: [Quick start](#quick-start) · [Get the runtime](#get-the-runtime) · [Start the API](#start-the-api) · [SDK License](#sdk-license) · [1:N identification](#1n-identification) · [Try it](#try-it) · [Setup on your own app](#setup-on-your-own-app) · [About SDK](#about-sdk)

## Quick start

- [ ] **Docker (recommended):** `docker pull faceplugin/face-recognition:latest`, then `docker run` — [Start the API](#option-a--docker)
- [ ] **Or native Linux:** download the runtime from [Releases](https://github.com/Faceplugin-ltd/FaceRecognition-Docker/releases/tag/latest) into `lib/cpu/`, then `./run.sh`
- [ ] Confirm it is running: `curl -s http://127.0.0.1:8092/api/health` (no license needed yet)
- [ ] [Contact us](#contact) with your machine code for a license key, then `POST /api/activate`
- [ ] Enroll your people, then **identify any face against all of them** — [1:N identification](#1n-identification)
- [ ] Try it with Postman, curl, or the Gradio demo on **9092** (`python3 demo.py`)

Docs: [doc.faceplugin.com](https://doc.faceplugin.com) · Try online: [Hugging Face Space](https://huggingface.co/spaces/FacePlugin-Ltd)

## Introduction

FacePlugin **Face Recognition** is a fully on-premise biometric engine for KYC, access control, attendance, and identity verification. Its headline capability is **1:N identification**: enroll your people once, then ask *who is this?* of any photo and get a ranked candidate list back — searched against a gallery that lives on your own server. It also does face detection (bounding box, landmarks, pose, attributes), ICAO-style quality analysis, template extraction, and **1:1 verification**.

The SDK runs as a REST API server through **Docker** (Linux, Windows, and macOS — Apple Silicon uses amd64 emulation), or natively on **Linux x86_64**. It runs on CPU only. This repository is self-contained; no other FacePlugin repository is required.

All processing runs on your own server, the gallery included, and **no** images or biometric data are ever sent to FacePlugin — built for banking, eKYC, and on-premise compliance workflows.

### Main Functionalities

| Feature | API |
| ------- | --- |
| **1:N identification** — search the enrollment gallery for every face in a photo | `POST /api/identify` · [how it works](#1n-identification) |
| **Gallery** — enroll, list, search, rename, delete, import a folder | `/api/persons` · `/api/faces` · `/api/persons/import` |
| Find people who look like a given face | `POST /api/similar` |
| **1:1 verification** (two images) | `POST /api/match` · `sdk.match` |
| Face detection (bounding box, landmarks, pose, attributes) | `POST /api/detect` · `sdk.detect` |
| Face quality analysis (ICAO-style checks) | `POST /api/quality` · `sdk.quality` |
| Face template extraction and similarity | `POST /api/feature` · `POST /api/similarity` |
| Health / machine code / activate | `GET /api/health` · `GET /api/machinecode` · `POST /api/activate` |
| License capabilities | `GET /api/licenseStatus` · `sdk.get_license_status` |

Every route also answers under `/api/v1` (for example `/api/v1/identify`), which is the path the Swagger docs and the Postman collection use.

### Product List

| Product | Repository |
| ------- | ---------- |
| Face Recognition — Windows | [FaceRecognition-Windows](https://github.com/Faceplugin-ltd/FaceRecognition-Windows) |
| **Face Recognition — Docker** | **[Docker Hub](https://hub.docker.com/r/faceplugin/face-recognition)** · **[FaceRecognition-Docker](https://github.com/Faceplugin-ltd/FaceRecognition-Docker)** **(this repo)** |
| Face Liveness Detection — Windows | [FaceLivenessDetection-Windows](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Windows) |
| Face Liveness Detection — Docker | [Docker Hub](https://hub.docker.com/r/faceplugin/face-liveness) · [FaceLivenessDetection-Docker](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Docker) |

### Related products

On-device face recognition + liveness apps for mobile:

- **Android** — [FaceRecognition-LivenessDetection-Android](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-Android)
- **iOS** — [FaceRecognition-LivenessDetection-iOS](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-iOS)

## Before you start

| Step | What you need |
| ---- | ------------- |
| 1 | Docker Engine, or native Linux x86_64 with glibc **2.38+** (for example Ubuntu 24.04) and Python 3.10+ |
| 2 | Docker: nothing to download, the runtime is inside the image. Native: the runtime in `lib/cpu/` — see [Get the runtime](#get-the-runtime) |
| 3 | Start **without** a license, copy the machine code, send it to FacePlugin ([contact](#contact)), then activate with your license key |

> ⚠️ **Docker and native installs have different machine codes.** Take the machine code from the same environment you will run in production.

### System requirements

| Item | Minimum | Recommended |
| ---- | ------- | ----------- |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 4 GB | 8 GB |
| OS (Docker) | Linux + Docker Engine | Ubuntu 22.04 / 24.04 |

## Get the runtime

Both editions of the engine are published in two places — the Docker image, and the release page for native Linux runs.

| Edition | Docker tag | Native runtime |
| ------- | ---------- | -------------- |
| **Lite** — the standard recognition engine | `faceplugin/face-recognition:latest` | Lite archive on [Releases](https://github.com/Faceplugin-ltd/FaceRecognition-Docker/releases/tag/latest) |
| **Plus** — the premium engine, for higher-accuracy deployments | `faceplugin/face-recognition:latest-plus` | Plus archive on [Releases](https://github.com/Faceplugin-ltd/FaceRecognition-Docker/releases/tag/latest) |

Both editions expose the same API and the same file layout, so you can switch between them without changing your integration.

**Docker** needs no download — the runtime is already inside the image.

**Native Linux** needs the runtime in `lib/cpu/`, which is empty on GitHub because the native libraries and model files are too large:

1. Clone the repo:

   ```bash
   git clone https://github.com/Faceplugin-ltd/FaceRecognition-Docker.git
   cd FaceRecognition-Docker
   ```

2. Download the Lite **or** Plus Linux archive from the release page above.
3. Extract it and place the files **directly** in `lib/cpu/`, not in a subfolder:

   ```text
   FaceRecognition-Docker/
   └── lib/
       └── cpu/
           ├── libFaceRecognitionSDK.so
           ├── libfacerec.so
           ├── far.fpk
           └── ... (remaining files from the archive)
   ```

> ⚠️ Move the files up out of any extracted folder, so you do not end up with `lib/cpu/SomeFolder/libFaceRecognitionSDK.so`.

## Start the API

You do **not** need a license to start the API. The server prints your machine code on startup, which you need to [get a license](#how-to-get-a-license). Product endpoints unlock after you activate.

### Option A — Docker

```bash
# Lite edition
sudo docker pull faceplugin/face-recognition:latest

# or the Plus edition
sudo docker pull faceplugin/face-recognition:latest-plus

sudo docker run -d --name faceplugin-face-recognition \
  --shm-size=2gb --privileged \
  -p 8092:8092 \
  -v /etc/machine-id:/etc/machine-id:ro \
  faceplugin/face-recognition:latest

sudo docker logs -f faceplugin-face-recognition
# Look for the machine code line in the logs
```

On Docker Desktop (macOS/Windows) omit the `/etc/machine-id` volume.

The repository also ships a [`docker-compose.yml`](docker-compose.yml) and a [`Dockerfile`](Dockerfile) if you prefer Compose or your own build.

**Several containers on one host:** to share one machine code and license across containers, see [the docs](https://doc.faceplugin.com/face-recognition-sdk/server-sdk/face-recognition-linux-sdk#run-multiple-containers).

### Option B — Native Linux (`./run.sh`)

```bash
pip3 install -r requirements.txt
./run.sh
```

API: **http://127.0.0.1:8092**. The machine code is printed in the terminal.

## SDK License

Licenses work **offline** and are tied to the machine code of the environment where the server runs. Offline cryptography is built into the SDK, so no OpenSSL install is needed.

### How to get a license

1. **Start the server** with Docker or `./run.sh` ([above](#start-the-api)). You do not need a license for the first start.
2. **Get your machine code.** It is printed in the startup log, or fetch it with `GET /api/machinecode`.
3. **Send the machine code to FacePlugin** ([contact us](#contact)). We reply with a license key for that machine code.
4. **Activate.** Save the license key to `license.txt` in the project root, replacing anything already in the file, then send it to the running server:

   ```bash
   curl -s -X POST http://127.0.0.1:8092/api/activate \
     -H 'Content-Type: text/plain' \
     --data-binary @license.txt
   ```

   In Docker this command is required, because a detached container does not re-read `license.txt` after it starts. The same command also works with `./run.sh`.

### License capabilities

This product exposes **recognition** APIs only.

```bash
curl -s http://127.0.0.1:8092/api/licenseStatus
```

## 1:N identification

This is what the product is for. **1:1 verification** answers *are these two photos the same person?*; **1:N identification** answers *who is this?* — it takes one photo, finds every face in it, and searches your whole enrollment gallery for each one, returning a ranked candidate list with similarity scores.

The gallery is a local database on the server (`data/`: SQLite plus face crops). It holds **persons**, and each person can have **several photos** — more photos per person make identification more reliable. Nothing leaves the server.

> In Docker, mount `data/` as a volume if you want the gallery to outlive the container.

### 1. Enroll your people

```bash
# One person per photo. Photos with the same name can be merged into one person.
curl -s -X POST http://127.0.0.1:8092/api/persons -F "image=@anna.jpg" -F "name=Anna Lee"
curl -s -X POST http://127.0.0.1:8092/api/persons -F "image=@anna2.jpg" -F "name=Anna Lee" -F "merge_by_name=true"

# A whole folder at once, subfolders included (set FACESDK_IMPORT_ROOT first).
# name_from=file takes each person's name from the file name, name_from=folder from its folder.
curl -s -X POST http://127.0.0.1:8092/api/persons/import -F "directory=staff" -F "name_from=file"
```

Enrollment checks:

| Check | What happens |
| ----- | ------------ |
| Duplicate | On by default (`duplicate_check` in `PATCH /api/settings`). Rejects a face that already matches another person (code `-7`, HTTP 409). Send `on_duplicate=add` to add the photo to that person, or `new` to enroll anyway. |
| Quality | Off by default to keep enrollment fast. Send `check_quality=true` to reject blurry, tiny, strongly turned, or low-quality faces with a reason (code `-8`, HTTP 422). Lighting and head tilt only warn. |

### 2. Identify

```bash
# Every face in the photo, top 5 candidates each
curl -s -X POST http://127.0.0.1:8092/api/identify -F "image=@probe.jpg" -F "top_k=5"

# Override the decision threshold for one request
curl -s -X POST http://127.0.0.1:8092/api/identify -F "image=@probe.jpg" -F "threshold=0.85"
```

Each face in the answer carries:

| Field | What it tells you |
| ----- | ----------------- |
| `status` | `identified` when the best candidate is at or above the decision threshold, otherwise `no_match` |
| best match | The person id, name, and similarity score |
| candidates | The ranked list, longest first, each with its own similarity |
| timing | How long the template extraction and the gallery search took |

The **decision threshold** is not hard-coded: send `threshold` with a request, change the stored default with `PATCH /api/settings`, or set it in the demo's Settings panel. `top_k` is per request. `POST /api/similar` answers the related question — *who in the gallery looks like this person?*

In Python, [`gallery.py`](gallery.py) is the same logic in process:

```python
import gallery

gallery.enroll(base64_image, name="Anna Lee")
result = gallery.identify(base64_probe, top_k=5)   # per-face status and candidates
```

Error codes: `-1` bad input · `-2` no face · `-4` import not allowed · `-5` too large · `-6` more than one face · `-7` duplicate · `-8` low quality · `-10` not found.

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/windows/demo-ui-identify.png" alt="FacePlugin Face Recognition demo — 1:N Identification tab with the probe photo, best match, and the ranked candidate list" width="900"/>
</p>

## Try it

### Health

```bash
curl -s http://127.0.0.1:8092/api/health
```

### Interactive API docs (Swagger)

Open **http://127.0.0.1:8092/docs** while the API runs. Every route, parameter, and error code is listed there, and you can send requests from the page. The OpenAPI 3 description is at `/api/v1/openapi.json`, for generating clients in any language.

### Security

By default the API listens on **127.0.0.1** (this machine only) and sends no CORS headers. In Docker it listens inside the container, and you publish the port yourself.

| Variable | Purpose |
| -------- | ------- |
| `FACESDK_BIND_HOST` | Listen address. Default `127.0.0.1` (this machine only). |
| `FACESDK_PORT` | Listen port (default 8092). |
| `FACESDK_CORS_ORIGINS` | Comma-separated browser origins allowed to call the API. |
| `FACESDK_IMPORT_ROOT` | Folder that `POST /api/persons/import` may read. Import is off when unset. |
| `FACESDK_MAX_UPLOAD_MB` | Largest request (default 20). |
| `FACESDK_THREADS` | Worker threads (default 8). |

### Postman

Import [`postman/FaceRecognition-API.postman_collection.json`](postman/FaceRecognition-API.postman_collection.json).

Default base URL: `http://127.0.0.1:8092`

### Demo UI (Gradio) — local only

The Docker image holds only the API server, not the demo UI. To see results in a browser, run the Gradio demo on your own machine with the API already running on port 8092.

```bash
pip3 install -r requirements-demo.txt
DEMO_PORT=9092 API_BASE=http://127.0.0.1:8092 python3 demo.py
```

Open **[http://127.0.0.1:9092](http://127.0.0.1:9092)**. Sample images, when included, are in `assets/examples/samples/`. The page header shows your license status, taken from `/api/licenseStatus`.

| Tab | What it shows |
| --- | ------------- |
| **1:N Identification** | Searches the gallery for every face in a photo, with a ranked candidate list and timing — see [1:N identification](#1n-identification) |
| **Gallery** | Enroll from files or a whole directory, search, rename, add or remove photos, find possible duplicates |
| **1:1 Verification** | Compares two photos: Match or No match against the decision threshold, plus the similarity of every face pair |
| **Face Detection** | Faces, pose, and attributes (age, emotion, glasses, mask, …), with a Result table and Raw JSON |
| **Quality Assessment** | ICAO-style face image quality checks |
| **API** | Endpoint list, curl examples, and links to the Swagger docs |

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/windows/demo-ui-detect.png" alt="FacePlugin Face Recognition demo — Face Detection tab with pose and attributes" width="900"/>
</p>

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/windows/demo-ui-quality.png" alt="FacePlugin Face Recognition demo — Quality Assessment tab with ICAO-style checks" width="900"/>
</p>

<p align="center">
 <img src="https://raw.githubusercontent.com/Faceplugin-ltd/faceplugin-assets/main/screenshots/face-recognition/windows/demo-ui-verify.png" alt="FacePlugin Face Recognition demo — 1:1 Verification tab with similarity score and decision threshold" width="900"/>
</p>

## Setup on your own app

Two ways to call the same engine. Full protocol: [https://doc.faceplugin.com](https://doc.faceplugin.com).

| Path | When to use |
| ---- | ----------- |
| **HTTP API** (Docker or `./run.sh`) | Any language. Keep the API running and send images to it. |
| **`sdk.py`** | Python on the **same** Linux machine as `lib/cpu/`. No HTTP hop. |

**HTTP (any language)**

1. Start the server and activate it once.
2. Enroll your people with `/api/persons`, then identify with `/api/identify` — [1:N identification](#1n-identification). For one-off comparisons use `/api/match`, and for the raw engine `/api/detect`, `/api/quality`, `/api/feature`, `/api/similarity`. Images go as base64 JSON or as form-data files.
3. Check every response for the SDK status code.

**Python in-process**

1. Copy [`sdk.py`](sdk.py) and `lib/cpu/` into your project.
2. Make sure `lib/cpu` is on `LD_LIBRARY_PATH` (that is all `run.sh` does).
3. Call `get_machine_code` → `activate` → `init_sdk`, then the recognition calls.

The Gradio demo (`demo.py`) is a local test UI. You do not need it in production.

## About SDK

[`sdk.py`](sdk.py) is the Python binding. Every call returns `0` on success, or an error code.

**1. Get the machine code and request a license**

```python
import sdk

print(sdk.get_machine_code())
```

**2. Activate, then initialize**

```python
ret = sdk.activate("license.txt")   # 0 on success
if ret == 0:
    ret = sdk.init_sdk()            # 0 on success

status = sdk.get_license_status()   # what the key unlocks
```

**3. Call the engine**

```python
sdk.detect(base64_image)                      # faces, pose, attributes
sdk.quality(base64_image)                     # ICAO-style quality checks
sdk.feature(base64_image)                     # face template
sdk.match(base64_image1, base64_image2)       # 1:1 verification
sdk.similarity(feature1_b64, feature2_b64)    # compare two templates
```

**4. Gallery and 1:N identification**

[`gallery.py`](gallery.py) keeps the enrolled people in SQLite and searches them with `sdk.feature` and `sdk.similarity`:

```python
import gallery

gallery.enroll(base64_image, name="Anna Lee")
result = gallery.identify(base64_probe, top_k=5)   # per-face status and candidates
```

HTTP endpoints (all under `/api`, also at `/api/v1`): `health`, `machinecode`, `licenseStatus`, `backend`, `activate`, `detect`, `quality`, `match`, `feature`, `similarity`, `identify`, `similar`, `settings`, `persons`, `persons/{id}`, `persons/import`, `faces/{id}`, `faces/{id}/image`. Full reference: `/docs`.

## Contact

<div align="left">
<a target="_blank" href="mailto:info@faceplugin.com"><img src="https://img.shields.io/badge/email-info@faceplugin.com-blue.svg?logo=gmail" alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://t.me/FacePluginSupport"><img src="https://img.shields.io/badge/telegram-@FacePluginSupport-blue.svg?logo=telegram" alt="Telegram @FacePluginSupport"></a>&emsp;
<a target="_blank" href="https://wa.me/+14692784822"><img src="https://img.shields.io/badge/whatsapp-faceplugin-blue.svg?logo=whatsapp" alt="faceplugin.com"></a>
</div>
