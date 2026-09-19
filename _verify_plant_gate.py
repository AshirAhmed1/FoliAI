"""Verify CLIP plant gate + hybrid pipeline decisions on synthetic images."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_gate import PLANT_GATE_NONPLANT_THRESHOLD, PlantLeafGate


def make_person_like() -> Image.Image:
    """Crude face/person-like synthetic (skin tones + oval head)."""
    img = Image.new("RGB", (320, 320), (180, 160, 140))
    draw = ImageDraw.Draw(img)
    draw.ellipse((90, 60, 230, 220), fill=(210, 170, 140))
    draw.rectangle((120, 200, 200, 320), fill=(40, 60, 120))
    return img


def make_car_like() -> Image.Image:
    img = Image.new("RGB", (320, 200), (200, 200, 205))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 80, 280, 150), fill=(30, 90, 180))
    draw.ellipse((60, 130, 110, 180), fill=(20, 20, 20))
    draw.ellipse((210, 130, 260, 180), fill=(20, 20, 20))
    return img


def make_desk_like() -> Image.Image:
    return Image.new("RGB", (320, 240), (90, 70, 50))


def make_building_like() -> Image.Image:
    img = Image.new("RGB", (320, 320), (135, 180, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 80, 260, 300), fill=(120, 120, 130))
    for y in range(100, 280, 35):
        for x in range(80, 240, 40):
            draw.rectangle((x, y, x + 20, y + 20), fill=(200, 220, 240))
    return img


def make_leaf_like() -> Image.Image:
    """Green leaf-shaped blob with vein-ish lines."""
    img = Image.new("RGB", (320, 320), (40, 90, 40))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 280, 280), fill=(50, 140, 55))
    draw.line((160, 50, 160, 270), fill=(30, 90, 30), width=3)
    for i in range(8):
        y = 70 + i * 25
        draw.line((160, y, 80 + i * 5, y + 20), fill=(35, 100, 35), width=2)
        draw.line((160, y, 240 - i * 5, y + 20), fill=(35, 100, 35), width=2)
    return img


def make_brown_leaf_like() -> Image.Image:
    """Diseased / autumn-ish brown leaf (should still pass gate)."""
    img = Image.new("RGB", (320, 320), (60, 50, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse((50, 50, 270, 270), fill=(140, 100, 45))
    draw.ellipse((90, 100, 150, 160), fill=(90, 50, 30))
    return img


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading plant gate on {device} (threshold={PLANT_GATE_NONPLANT_THRESHOLD})...")
    gate = PlantLeafGate(device)

    def make_pet_like() -> Image.Image:
        """Crude animal/pet-like blob (fur tones + ears) — not a flat color field."""
        img = Image.new("RGB", (320, 320), (210, 200, 190))
        draw = ImageDraw.Draw(img)
        draw.ellipse((80, 100, 240, 260), fill=(160, 110, 70))
        draw.ellipse((70, 60, 130, 130), fill=(150, 100, 60))
        draw.ellipse((190, 60, 250, 130), fill=(150, 100, 60))
        draw.ellipse((120, 160, 145, 185), fill=(20, 20, 20))
        draw.ellipse((175, 160, 200, 185), fill=(20, 20, 20))
        return img

    non_plant_cases = [
        ("person_like", make_person_like()),
        ("car_like", make_car_like()),
        ("desk_like", make_desk_like()),
        ("building_like", make_building_like()),
        ("pet_like", make_pet_like()),
    ]
    plant_cases = [
        ("green_leaf", make_leaf_like()),
        ("brown_leaf", make_brown_leaf_like()),
        ("solid_green_foliage", Image.new("RGB", (300, 300), (34, 120, 40))),
    ]
    ambiguous_cases = [
        ("grass_like", Image.new("RGB", (300, 120), (60, 130, 50))),
        ("sky", Image.new("RGB", (300, 300), (120, 180, 230))),
    ]

    rejected = 0
    print("\n=== OBVIOUS NON-PLANT ===")
    for name, img in non_plant_cases:
        r = gate.classify(img)
        ok = r.decision == "non_plant"
        rejected += int(ok)
        print(
            f"  {name:16} decision={r.decision:10} "
            f"plant={r.plant_score:.3f} non_plant={r.non_plant_score:.3f} "
            f"{'OK' if ok else 'MISS'}"
        )

    passed = 0
    false_reject = 0
    print("\n=== PLANT / LEAF ===")
    for name, img in plant_cases:
        r = gate.classify(img)
        ok = r.is_plant  # plant or uncertain both pass
        passed += int(ok)
        false_reject += int(not ok)
        print(
            f"  {name:16} decision={r.decision:10} "
            f"plant={r.plant_score:.3f} non_plant={r.non_plant_score:.3f} "
            f"{'PASS' if ok else 'FALSE_REJECT'}"
        )

    print("\n=== AMBIGUOUS ===")
    for name, img in ambiguous_cases:
        r = gate.classify(img)
        print(
            f"  {name:16} decision={r.decision:10} "
            f"plant={r.plant_score:.3f} non_plant={r.non_plant_score:.3f}"
        )

    print("\n=== SUMMARY ===")
    print(f"non-plant rejected: {rejected}/{len(non_plant_cases)}")
    print(f"leaf passed: {passed}/{len(plant_cases)}")
    print(f"false rejections: {false_reject}")
    print(f"threshold: {PLANT_GATE_NONPLANT_THRESHOLD}")

    # Hybrid untouched smoke check
    import main as app_main

    app_main.device = device
    app_main.plant_gate = gate
    app_main.model_v3 = app_main.load_model(app_main.V3_MODEL_PATH, "v3")
    app_main.model_v5 = app_main.load_model(app_main.V5_MODEL_PATH, "v5")

    # Person-like should short-circuit without needing hybrid for response shape
    person = make_person_like()
    g = gate.classify(person)
    assert g.decision == "non_plant" or g.decision == "uncertain"
    # Desk often strong non-plant
    desk = gate.classify(make_desk_like())
    leaf = gate.classify(make_leaf_like())
    assert leaf.is_plant, "green leaf must pass plant gate"
    print(
        f"desk decision={desk.decision} leaf decision={leaf.decision} "
        f"(assert leaf passes)"
    )
    print("OK plant gate verification finished")


if __name__ == "__main__":
    main()
