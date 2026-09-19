"""HTTP smoke test for three-state /predict responses."""
from __future__ import annotations

import io
import json
import urllib.request
from PIL import Image, ImageDraw


def post_image(name: str, img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()
    boundary = "----foliai"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    body = header + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode())


def make_person() -> Image.Image:
    img = Image.new("RGB", (320, 320), (180, 160, 140))
    draw = ImageDraw.Draw(img)
    draw.ellipse((90, 60, 230, 220), fill=(210, 170, 140))
    draw.rectangle((120, 200, 200, 320), fill=(40, 60, 120))
    return img


def make_leaf() -> Image.Image:
    img = Image.new("RGB", (320, 320), (40, 90, 40))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 280, 280), fill=(50, 140, 55))
    draw.line((160, 50, 160, 270), fill=(30, 90, 30), width=3)
    for i in range(8):
        y = 70 + i * 25
        draw.line((160, y, 80 + i * 5, y + 20), fill=(35, 100, 35), width=2)
        draw.line((160, y, 240 - i * 5, y + 20), fill=(35, 100, 35), width=2)
    return img


def main() -> None:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as r:
        health = json.loads(r.read().decode())
    print("HEALTH", health)
    assert health.get("plant_gate_loaded") is True
    assert health.get("v3_loaded") is True
    assert health.get("v5_loaded") is True

    person = post_image("person", make_person())
    print("PERSON", {k: person[k] for k in ("reason", "matched", "input_valid", "predictions", "message")})
    assert person["reason"] == "non_plant"
    assert person["input_valid"] is False
    assert person["predictions"] == []
    assert person["matched"] is False

    leaf = post_image("green_leaf", make_leaf())
    print(
        "LEAF",
        {
            k: leaf[k]
            for k in ("reason", "matched", "input_valid", "confidence", "class_name")
        },
        "n_preds=",
        len(leaf["predictions"]),
    )
    assert leaf["input_valid"] is True
    assert leaf["reason"] in (None, "low_confidence")
    assert len(leaf["predictions"]) == 3
    print("HTTP three-state checks OK")


if __name__ == "__main__":
    main()
