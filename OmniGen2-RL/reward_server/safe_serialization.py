"""
Safe serialization utilities for reward server communication.

Replaces pickle-based serialization with JSON + base64-encoded images
to prevent arbitrary code execution via deserialization attacks (CWE-502).
"""

import io
import json
import base64
from typing import Any

from PIL import Image

_IMAGE_MARKER = "__pil_image__"
_TUPLE_MARKER = "__tuple__"


def _encode(obj: Any) -> Any:
    if isinstance(obj, Image.Image):
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        return {_IMAGE_MARKER: base64.b64encode(buf.getvalue()).decode("ascii")}
    if isinstance(obj, tuple):
        return {_TUPLE_MARKER: [_encode(item) for item in obj]}
    if isinstance(obj, list):
        return [_encode(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    raise TypeError(f"Unsupported type for safe serialization: {type(obj)}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if _IMAGE_MARKER in obj:
            return Image.open(io.BytesIO(base64.b64decode(obj[_IMAGE_MARKER])))
        if _TUPLE_MARKER in obj:
            return tuple(_decode(item) for item in obj[_TUPLE_MARKER])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(item) for item in obj]
    return obj


def safe_dumps(data: Any) -> bytes:
    """Serialize data safely using JSON with base64-encoded images."""
    return json.dumps(_encode(data), separators=(",", ":")).encode("utf-8")


def safe_loads(data: bytes) -> Any:
    """Deserialize data safely from JSON with base64-encoded images."""
    return _decode(json.loads(data))
