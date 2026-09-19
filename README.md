# 🌿 FoliAI

**FoliAI** is a full-stack plant disease classification application that analyzes leaf images and predicts common diseases affecting **bell pepper, potato, and tomato plants**.

Users can upload an existing image or capture a new photo using their device camera. FoliAI processes the image through a **class-specific ensemble of two fine-tuned ResNet-18 models**, returns the most likely class, displays the top three predictions with confidence scores, and rejects uncertain images rather than always forcing a diagnosis.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black)
![TypeScript](https://img.shields.io/badge/TypeScript-Frontend-3178C6)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

FoliAI was built to provide a simple interface for experimenting with computer-vision-based plant disease recognition.

The application currently supports **15 healthy and disease classes across 3 crops**:

- Bell pepper
- Potato
- Tomato

A user provides a close-up image of a leaf, and FoliAI:

1. validates the uploaded image,
2. resizes and preprocesses it,
3. runs the image through both the **v3 and v5 ResNet-18 models**,
4. combines their predictions using a class-specific hybrid routing system,
5. applies confidence calibration,
6. returns the most likely class and top-3 alternatives,
7. rejects the prediction if confidence is too low.

---

# ✨ Features

### Image upload

Users can select an existing image from their device or drag and drop one directly into the application.

Supported formats include:

- JPEG
- PNG
- WebP
- GIF

Maximum upload size:

**10 MB**

Images are resized client-side before being uploaded to reduce transfer size and inference latency.

---

### 📷 Camera capture

FoliAI also supports capturing an image directly from the device running the website.

Users can choose:

- **Upload image** — select an existing image
- **Take photo** — use the current device's camera

On supported mobile devices, the browser can open the device camera directly.

On desktops and laptops with a webcam, camera access can be granted through the browser and a leaf image can be captured without first saving it manually.

Camera images use the same validation, preprocessing, and prediction pipeline as uploaded images.

---

### Top-3 predictions

Instead of returning only one class, FoliAI displays the three highest-scoring possibilities.

Example:

```text
1. Tomato Late Blight       82.4%
2. Tomato Early Blight      10.8%
3. Tomato Bacterial Spot     4.1%
```

This provides more context when visually similar diseases receive competing scores.

---

### Confidence-based rejection

Image classifiers can produce high-confidence predictions even when an image does not belong to any class they were trained to recognize.

FoliAI therefore does not blindly return a disease for every image.

The model probabilities are temperature-scaled using:

```text
T = 2.5
```

If the final hybrid prediction has less than:

```text
60% confidence
```

the application returns:

```text
No clear match
```

instead of presenting the highest-scoring class as a reliable result.

The interface still displays the closest matches so the user can see how the model interpreted the image.

---

# 🧠 Model Architecture

FoliAI currently uses a **v3 + v5 hybrid ensemble**.

Both models are based on:

```text
ResNet-18
ImageNet pretrained weights
15-class classification head
224 × 224 input images
```

The two models were trained during different stages of dataset development and showed different strengths when evaluated on real-world plant images.

Rather than replacing v3 entirely with v5, the production system combines them at the **class probability level**.

---

## v3 model

The original v3 model was trained using approximately:

```text
24,245 images
```

from:

- PlantVillage
- PlantDoc
- targeted Tomato Leaf data

Its original internal test performance was:

```text
Accuracy: 97.53%
Macro F1: ~0.97
Weighted F1: ~0.97
```

v3 remains particularly strong for several of the original classes.

---

## v5 model

v5 expanded the training corpus to:

```text
25,667 images
```

with additional real-world examples from sources including:

- PlantWild
- uncontrolled-environment potato imagery

v5 showed stronger performance on several disease classes during external testing.

---

# 🔀 v3 + v5 Hybrid Inference

Every valid image is passed through **both models**.

The logits from each model are converted into probabilities using:

```python
softmax(logits / 2.5)
```

FoliAI then builds a new 15-class probability vector.

### Classes sourced from v5

v5 provides the probability for:

- Potato Late Blight
- Tomato Early Blight
- Tomato Late Blight
- Tomato Mosaic Virus

### Potato Healthy

For:

```text
Potato Healthy
```

FoliAI averages the v3 and v5 probabilities.

### Remaining classes

All other class probabilities are taken from v3.

The resulting probability vector is then normalized so that all 15 probabilities sum to 1.

The final:

- predicted class
- confidence
- top-3 predictions
- 60% rejection decision

are all calculated from this **hybrid probability distribution**.

---

# 🌱 Supported Classes

FoliAI currently predicts the following 15 classes.

### Bell Pepper

1. Bacterial Spot
2. Healthy

### Potato

3. Early Blight
4. Late Blight
5. Healthy

### Tomato

6. Bacterial Spot
7. Early Blight
8. Late Blight
9. Leaf Mold
10. Septoria Leaf Spot
11. Two-Spotted Spider Mite
12. Target Spot
13. Tomato Yellow Leaf Curl Virus
14. Tomato Mosaic Virus
15. Healthy

The canonical internal class ordering is:

```text
0  Pepper__bell___Bacterial_spot
1  Pepper__bell___healthy
2  Potato___Early_blight
3  Potato___Late_blight
4  Potato___healthy
5  Tomato_Bacterial_spot
6  Tomato_Early_blight
7  Tomato_Late_blight
8  Tomato_Leaf_Mold
9  Tomato_Septoria_leaf_spot
10 Tomato_Spider_mites_Two_spotted_spider_mite
11 Tomato__Target_Spot
12 Tomato__Tomato_YellowLeaf__Curl_Virus
13 Tomato__Tomato_mosaic_virus
14 Tomato_healthy
```

---

# 🖥️ Application Architecture

```text
                         ┌─────────────────┐
                         │   Leaf Image    │
                         └────────┬────────┘
                                  │
                    Upload or camera capture
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Next.js Frontend│
                         └────────┬────────┘
                                  │
                         Resize / compress
                                  │
                                  ▼
                         POST /predict
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ FastAPI Backend │
                         └────────┬────────┘
                                  │
                         Image validation
                                  │
                         Resize to 224×224
                                  │
                                  ▼
                  ┌───────────────┴───────────────┐
                  │                               │
                  ▼                               ▼
          ┌───────────────┐               ┌───────────────┐
          │ ResNet-18 v3  │               │ ResNet-18 v5  │
          └───────┬───────┘               └───────┬───────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                           T = 2.5 softmax
                                  │
                                  ▼
                       Class-specific routing
                                  │
                                  ▼
                      Hybrid probability vector
                                  │
                                  ▼
                Top-1 + confidence + top-3 results
                                  │
                         confidence >= 60%?
                           /             \
                         yes              no
                          │                │
                    Likely match      No clear match
```

---

# 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Machine Learning | PyTorch, torchvision, ResNet-18 |
| Image Processing | Pillow, torchvision transforms |
| Backend | FastAPI, Python, Uvicorn |
| Frontend | Next.js, React, TypeScript |
| Styling | Tailwind CSS |
| Model Serving | Local PyTorch inference |

---

# 📁 Project Structure

A simplified project structure looks like:

```text
FoliAI/
│
├── main.py
├── requirements.txt
├── _verify_hybrid.py
│
├── model/
│   ├── best_model.pth
│   ├── best_model_v5.pth
│   ├── best_model_v1_backup.pth
│   └── best_model_v2_backup.pth
│
├── frontend/
│   ├── app/
│   ├── public/
│   ├── package.json
│   └── .env.local
│
└── README.md
```

The two checkpoints required for the current production hybrid are:

```text
model/best_model.pth
model/best_model_v5.pth
```

`best_model.pth` is v3.

`best_model_v5.pth` is v5.

---

# 🚀 Running FoliAI Locally

FoliAI currently runs as a local web application consisting of:

```text
FastAPI backend   → http://localhost:8000
Next.js frontend  → http://localhost:3000
```

Both servers must be running to use the full application.

---

## Prerequisites

Install:

- Python 3.9+
- Node.js 18+
- npm
- Git

The required model checkpoints must also exist in:

```text
model/best_model.pth
model/best_model_v5.pth
```

---

## 1. Clone the repository

```bash
git clone <repository-url>
cd FoliAI
```

If you are viewing this repository on GitHub, you can obtain the clone URL from:

```text
Code → HTTPS
```

---

## 2. Install Python dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Start the FastAPI backend

From the project root:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI's interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

## 4. Configure the frontend

Enter the frontend directory:

```bash
cd frontend
```

Create:

```text
.env.local
```

with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

If this file already exists, no additional configuration is required for normal local use.

---

## 5. Install frontend dependencies

Inside `frontend/`:

```bash
npm install
```

---

## 6. Start the frontend

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

in your browser.

You should now see the FoliAI classifier.

---

# 📸 How to Use FoliAI

Once both servers are running:

### Option 1: Upload an image

1. Open `http://localhost:3000`.
2. Click **Upload image**.
3. Select a leaf image from your computer.
4. Wait while FoliAI analyzes the image.
5. Review the result and confidence score.
6. Review the top-3 closest classes.
7. Click **Upload another image** to classify another leaf.

You can also drag and drop an image directly into the upload area.

---

### Option 2: Take a photo

1. Click **Take photo**.
2. Allow camera access if your browser asks for permission.
3. Capture a clear photo of the leaf.
4. Confirm/use the captured image.
5. FoliAI processes it through the same prediction pipeline as an uploaded file.

For best results:

- keep the leaf in focus,
- use good lighting,
- keep the leaf relatively close to the camera,
- avoid excessive background clutter,
- try to keep the relevant leaf filling a large portion of the frame.

---

# 🔌 API Reference

## `POST /predict`

Runs FoliAI inference on an uploaded image.

### Request

Multipart form request containing:

```text
file
```

Accepted images:

```text
JPEG
PNG
WebP
GIF
```

Maximum upload size:

```text
10 MB
```

### Example successful response

```json
{
  "class_name": "Tomato_Late_blight",
  "confidence": 0.87,
  "matched": true,
  "message": null,
  "predictions": [
    {
      "class_name": "Tomato_Late_blight",
      "confidence": 0.87
    },
    {
      "class_name": "Tomato_Early_blight",
      "confidence": 0.09
    },
    {
      "class_name": "Tomato_healthy",
      "confidence": 0.02
    }
  ]
}
```

### Low-confidence response

If the top hybrid confidence is below 60%, FoliAI reports that no supported class was matched confidently.

Example:

```json
{
  "class_name": null,
  "confidence": 0.34,
  "matched": false,
  "message": "Image does not clearly match any known plant disease class.",
  "predictions": [
    {
      "class_name": "Tomato_Early_blight",
      "confidence": 0.34
    },
    {
      "class_name": "Tomato_Late_blight",
      "confidence": 0.21
    },
    {
      "class_name": "Potato___Early_blight",
      "confidence": 0.12
    }
  ]
}
```

---

## `GET /health`

Reports whether the backend and ensemble models are available.

The current backend identifies the production model as the:

```text
v3+v5 hybrid
```

Example response may include:

```json
{
  "status": "ok",
  "ensemble": "v3+v5_hybrid"
}
```

---

# 🔐 Input Validation and Security

The backend does not rely only on a file extension or browser-provided MIME type.

Uploaded files are:

- capped at 10 MB,
- read with a bounded upload size,
- decoded and validated as real image data,
- rejected if the image cannot be decoded,
- handled using generic API error responses rather than exposing Python tracebacks.

CORS is restricted to the configured frontend origin rather than allowing unrestricted wildcard access.

---

# ⚡ Frontend Performance

Before an image is sent to the backend, the frontend reduces unnecessary transfer size by resizing large images.

Images are generally:

```text
Maximum dimension: 800 px
JPEG quality: ~85%
```

The frontend also includes:

- a 30-second request timeout,
- disabled controls while inference is running,
- loading feedback,
- drag-and-drop state,
- camera capture,
- top-3 confidence bars,
- upload reset controls,
- object URL cleanup,
- keyboard-accessible upload controls.

---

# 🧪 Verifying the Hybrid

The repository includes:

```bash
python _verify_hybrid.py
```

This performs lightweight checks on the ensemble implementation, including:

- v3 loading,
- v5 loading,
- 15-class output shape,
- class-specific routing,
- hybrid probability normalization,
- top-3 behavior.

---

# 📊 Model Limitations

FoliAI is a computer vision project and should not be interpreted as a guaranteed plant diagnosis system.

Current limitations include:

- only bell pepper, potato, and tomato are supported,
- unsupported crops can still resemble trained classes,
- blurry or distant images can reduce reliability,
- cluttered backgrounds can affect predictions,
- lighting and camera conditions can differ substantially from training data,
- some visually similar diseases remain difficult to distinguish,
- healthy potato remains a particularly difficult real-world class.

The 60% confidence gate reduces some misleading predictions, but confidence alone cannot guarantee correctness.

---

# 🛣️ Future Improvements

Potential future work includes:

- collecting more uncontrolled field images,
- improving healthy-potato generalization,
- introducing a crop-identification stage before disease classification,
- evaluating a hierarchical crop → disease architecture,
- expanding beyond bell pepper, potato, and tomato,
- Grad-CAM visualizations for model interpretability,
- larger independent external benchmarks,
- improved confidence calibration,
- hosted/cloud deployment.

---

# 🤝 Contributing

Issues and pull requests are welcome.

For substantial changes, please open an issue first to discuss the proposed modification.

---

# 📄 License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.

---

## FoliAI

**Computer vision for plant disease recognition using PyTorch, FastAPI, Next.js, and a class-specific ResNet-18 ensemble.**
