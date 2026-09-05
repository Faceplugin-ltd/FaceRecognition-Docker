"""ctypes bindings to libFaceRecognitionSDK.so (Face Recognition).

Native libs live in lib/cpu/.
"""

from __future__ import annotations

import base64
import ctypes
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib" / "cpu"
SO = LIB / "libFaceRecognitionSDK.so"

os.chdir(ROOT)
os.environ["LD_LIBRARY_PATH"] = str(LIB) + (
    (":" + os.environ["LD_LIBRARY_PATH"]) if os.environ.get("LD_LIBRARY_PATH") else ""
)

if not SO.is_file():
    raise FileNotFoundError(
        f"missing {SO} — put Drive files in lib/cpu/ "
        "(export copies SDK lib/ → App lib/cpu/)"
    )

_dll = ctypes.cdll.LoadLibrary(str(SO))

_dll.FaceSDK_initSDK.restype = ctypes.c_int
_dll.FaceSDK_initSDK.argtypes = []
_dll.FaceSDK_activate.restype = ctypes.c_int
_dll.FaceSDK_activate.argtypes = [ctypes.c_char_p]
_dll.FaceSDK_getMachineCode.restype = ctypes.c_int
_dll.FaceSDK_getMachineCode.argtypes = [ctypes.c_char_p]
_dll.FaceSDK_detect.restype = ctypes.c_int
_dll.FaceSDK_detect.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
_dll.FaceSDK_quality.restype = ctypes.c_int
_dll.FaceSDK_quality.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
_dll.FaceSDK_match.restype = ctypes.c_int
_dll.FaceSDK_match.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_bool,
]
_dll.FaceSDK_getFeature.restype = ctypes.c_int
_dll.FaceSDK_getFeature.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_dll.FaceSDK_getSimilarity.restype = ctypes.c_float
_dll.FaceSDK_getSimilarity.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]

_MC_BUF = 768
_OUT_MIN = 1_000_000
_OUT_MAX = 16 * 1024 * 1024
_MAX_IMG = 8 * 1024 * 1024


def _b(v) -> bytes:
    if v is None:
        raise ValueError("required")
    if isinstance(v, bytes):
        return v
    return str(v).encode("utf-8")


def _out(*parts: bytes):
    n = min(max(_OUT_MIN, sum(len(p) for p in parts) * 2 + 262144), _OUT_MAX)
    return ctypes.create_string_buffer(n)


def _strip(raw: bytes) -> bytes:
    text = raw.decode("utf-8", errors="ignore").strip()
    if text.startswith("data:") and "base64," in text:
        text = text.split("base64,", 1)[1]
    return text.encode("utf-8")


def _fit_b64(image_b64: str, max_bytes: int = _MAX_IMG) -> bytes:
    raw = _strip(_b(image_b64))
    if len(raw) <= max_bytes:
        return raw
    from PIL import Image

    data = base64.b64decode(raw, validate=False)
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    quality, scale, last = 85, 1.0, raw
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    for _ in range(12):
        cand = img
        if scale < 1.0:
            cand = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                resample,
            )
        buf = io.BytesIO()
        cand.save(buf, format="JPEG", quality=quality, optimize=True)
        last = base64.b64encode(buf.getvalue())
        if len(last) <= max_bytes:
            return last
        if quality > 40:
            quality -= 10
        else:
            scale *= 0.75
            quality = 75
    return last


def get_machine_code() -> str:
    """Return FPMC1.… machine code."""
    buf = ctypes.create_string_buffer(_MC_BUF)
    _dll.FaceSDK_getMachineCode(buf)
    return buf.value.decode("utf-8", errors="replace")


def activate(license_path: str) -> int:
    """Activate with path to license.txt / license.dat, or an FP1.… key string."""
    return int(_dll.FaceSDK_activate(_b(license_path)))


def init_sdk() -> int:
    return int(_dll.FaceSDK_initSDK())


def detect(image: str, crop_image: bool = False) -> str:
    image_b = _fit_b64(image)
    out = _out(image_b)
    _dll.FaceSDK_detect(image_b, out, bool(crop_image))
    return out.value.decode("utf-8", errors="replace")


def quality(image: str, crop_image: bool = False) -> str:
    image_b = _fit_b64(image)
    out = _out(image_b)
    _dll.FaceSDK_quality(image_b, out, bool(crop_image))
    return out.value.decode("utf-8", errors="replace")


def match(image1: str, image2: str, crop_image: bool = False) -> str:
    a = _fit_b64(image1, _MAX_IMG // 2)
    b = _fit_b64(image2, _MAX_IMG // 2)
    out = _out(a, b)
    _dll.FaceSDK_match(a, b, out, bool(crop_image))
    return out.value.decode("utf-8", errors="replace")


def feature(image: str) -> str:
    image_b = _fit_b64(image)
    out = _out(image_b)
    _dll.FaceSDK_getFeature(image_b, out)
    return out.value.decode("utf-8", errors="replace")


def similarity(feature1: str, feature2: str) -> str:
    raw1 = base64.b64decode(_b(feature1))
    raw2 = base64.b64decode(_b(feature2))
    if len(raw1) != len(raw2) or not raw1:
        raise ValueError("feature length mismatch")
    value = _dll.FaceSDK_getSimilarity(raw1, raw2, len(raw1))
    return json.dumps({"similarity": value})
