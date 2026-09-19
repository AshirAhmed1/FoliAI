"""Smoke-test capture-session endpoints (no phone required)."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from PIL import Image

BASE = "http://127.0.0.1:8000"


def req(method: str, path: str, data: bytes | None = None, headers: dict | None = None):
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            body = resp.read()
            return resp.status, body, dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


def main() -> None:
    status, body, _ = req("POST", "/capture-session")
    assert status == 200, body
    created = json.loads(body)
    sid = created["session_id"]
    assert created["status"] == "waiting"
    print("created", sid)

    status, body, _ = req("GET", f"/capture-session/{sid}")
    assert status == 200
    assert json.loads(body)["status"] == "waiting"

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (20, 140, 40)).save(buf, format="JPEG")
    raw = buf.getvalue()
    boundary = "----B"
    form = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="leaf.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()

    status, body, _ = req(
        "POST",
        f"/capture-session/{sid}/upload",
        data=form,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert status == 200, body
    assert json.loads(body)["status"] == "uploaded"
    print("uploaded")

    # Second upload rejected
    status, body, _ = req(
        "POST",
        f"/capture-session/{sid}/upload",
        data=form,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert status == 409, body
    print("second upload rejected", status)

    status, body, headers = req("GET", f"/capture-session/{sid}/image")
    assert status == 200, body
    assert body[:2] == b"\xff\xd8"  # JPEG SOI
    print("image fetched", len(body), headers.get("Content-Type"))

    # Consumed — no longer downloadable
    status, body, _ = req("GET", f"/capture-session/{sid}/image")
    assert status in (404, 409), body
    print("consumed protected", status)

    # Unknown session
    status, body, _ = req("GET", "/capture-session/does-not-exist")
    assert status == 404
    print("unknown rejected")

    # Delete fresh session
    status, body, _ = req("POST", "/capture-session")
    sid2 = json.loads(body)["session_id"]
    status, body, _ = req("DELETE", f"/capture-session/{sid2}")
    assert status == 200
    status, body, _ = req("GET", f"/capture-session/{sid2}")
    assert status == 404
    print("delete ok")

    print("ALL CAPTURE SESSION CHECKS PASSED")


if __name__ == "__main__":
    main()
