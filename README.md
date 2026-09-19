# FoliAI


# 🌿 FoliAI

A plant disease classifier: upload a photo of a leaf, get back the most likely disease (or a clean bill of health) with a confidence score — backed by a fine-tuned ResNet-18 model served over FastAPI, with a Next.js frontend.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688.svg)
![Next.js](https://img.shields.io/badge/Next.js-frontend-black.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## What it does

- Classifies leaf images into **15 classes** across pepper, potato, and tomato plants — bacterial spot, early/late blight, leaf mold, spider mites, mosaic virus, and healthy, among others
- Returns the **top-3 predictions** with confidence scores, not just a single label
- **Rejects low-confidence predictions** instead of forcing a guess — if the top prediction is below a 60% confidence threshold, the API reports "no clear match" rather than a false positive

## Demo

> _Add a screenshot or GIF of the upload → prediction flow here._

## How it works

### Two-stage inference

1. **Plant / leaf scope gate** (OpenCLIP ViT-B-32) — a semantic input gate that rejects *obvious* non-plant images (people, cars, furniture, etc.) before disease classification. It compares the image against plant/leaf and non-plant text concepts; uncertain cases are allowed through. False rejection of real leaves is treated as worse than a later low-confidence disease result. First startup downloads OpenAI CLIP weights (~338MB) into the local Hugging Face cache.
2. **v3 + v5 class-specific hybrid** — two ResNet-18 disease models whose class probabilities are combined with frozen routing, temperature scaling (T = 2.5), and a 60% confidence gate.

FoliAI can therefore distinguish:

- **Non-plant / out of scope** — no disease top-3 shown
- **Plant-like, low disease confidence** — closest disease matches shown
- **Likely disease / healthy match** — top prediction + top-3

The plant detector is probabilistic and not perfect.

### Backend (`main.py`, FastAPI)

- ResNet-18 v3 + v5 hybrid (`model/best_model.pth`, `model/best_model_v5.pth`)
- OpenCLIP plant/leaf gate loaded once at startup (weights cached by Hugging Face / open_clip after first download)
- **Temperature-scaled softmax** (T = 2.5) and **60% disease confidence threshold**
- **Input validation**: PIL decode, 10MB cap, generic error messages
- **CORS** locked to the frontend origin

### Frontend (`frontend/`, Next.js + React)

- Upload, local camera, and drag-and-drop
- Three result states: likely match, no clear match (plant), not a supported leaf image
- Client-side resize before upload

## Tech stack

| Layer | Tools |
|---|---|
| ML | PyTorch, torchvision (ResNet-18), OpenCLIP (ViT-B-32 plant gate) |
| Backend | FastAPI, Pillow, python-multipart |
| Frontend | Next.js, React, TypeScript, Tailwind CSS |

## Getting started

### Prerequisites

- Python 3.9+
- Node.js 18+
- A trained model checkpoint at `model/best_model.pth`

### Backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

By default the frontend expects the API at `http://localhost:8000` — override with the `NEXT_PUBLIC_API_URL` environment variable.

## API reference

### `POST /predict`

Multipart form upload with a `file` field (JPEG, PNG, WebP, or GIF, max 10MB).

**Response:**

```json
{
  "class_name": "Tomato_Late_blight",
  "confidence": 0.87,
  "matched": true,
  "input_valid": true,
  "reason": null,
  "message": null,
  "predictions": [
    { "class_name": "Tomato_Late_blight", "confidence": 0.87 },
    { "class_name": "Tomato_Early_blight", "confidence": 0.09 },
    { "class_name": "Tomato_healthy", "confidence": 0.02 }
  ]
}
```

Non-plant / out-of-scope example:

```json
{
  "class_name": null,
  "confidence": null,
  "matched": false,
  "input_valid": false,
  "reason": "non_plant",
  "message": "This image does not appear to contain a supported plant leaf. ...",
  "predictions": []
}
```

### `GET /health`

Reports whether the plant gate and disease models are loaded.

```json
{
  "status": "ok",
  "plant_gate_loaded": true,
  "v3_loaded": true,
  "v5_loaded": true,
  "ensemble": "v3+v5_hybrid"
}
```

## Supported classes

<details>
<summary>Click to expand — 15 classes across 3 crops</summary>

- **Pepper**: bacterial spot, healthy
- **Potato**: early blight, late blight, healthy
- **Tomato**: bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, mosaic virus, yellow leaf curl virus, healthy

</details>

## Roadmap

- [ ] Model soup / weight averaging across multiple fine-tuning runs for improved accuracy and robustness
- [ ] Grad-CAM visualization to verify predictions are grounded in actual lesion features, not background artifacts
- [ ] Expand class coverage to additional crops

## License

MIT — see [LICENSE](LICENSE) for details.

## Contributing

Issues and pull requests welcome. For major changes, please open an issue first to discuss what you'd like to change.
