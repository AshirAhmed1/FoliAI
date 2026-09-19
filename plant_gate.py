"""
Semantic plant/leaf scope gate using OpenCLIP (ViT-B-32).

This is a first-stage OOD filter BEFORE the v3+v5 disease ensemble.
It is probabilistic — reject only on strong non-plant evidence.
False rejection of real leaves is worse than letting ambiguous images through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger("foliai.plant_gate")

# Conservative non-plant mass threshold used together with an argmax-prompt check.
# Reject only when BOTH:
#   1) summed non-plant prompt probability >= this threshold, AND
#   2) the single highest-probability text prompt is a NON-PLANT concept.
#
# Rationale: diseased/autumn leaves can still put substantial mass on non-plant
# prompts (textures, brown objects), but their *top* CLIP match is usually a
# plant/leaf prompt. Requiring both conditions avoids false leaf rejections
# while still catching obvious people / vehicles / buildings / empty scenes.
#
# Tuned on synthetic probes with OpenCLIP ViT-B-32 (openai weights).
# Prefer false accepts (→ disease "no clear match") over false rejects of leaves.
PLANT_GATE_NONPLANT_THRESHOLD = 0.62

PLANT_PROMPTS = [
    "a close-up photograph of a plant leaf",
    "a photograph of a diseased plant leaf",
    "a photograph of a healthy plant leaf",
    "a photograph of a crop plant",
    "a leaf on a plant",
    # Extra plant concepts so brown/diseased foliage is not dominated by
    # "household object" / "no plant" prompts alone.
    "a brown or yellowing diseased plant leaf",
    "a close-up of foliar plant tissue",
]

NON_PLANT_PROMPTS = [
    "a photograph of a person",
    "a photograph of an animal",
    "a photograph of a vehicle",
    "a photograph of a building",
    "a photograph of furniture",
    "a photograph of a household object",
    "a photograph with no plant or leaf",
    "a random non-plant photograph",
]


@dataclass
class PlantGateResult:
    is_plant: bool
    plant_score: float
    non_plant_score: float
    decision: str  # "plant" | "non_plant" | "uncertain"


class PlantLeafGate:
    """Zero-shot CLIP plant/leaf gate with cached text embeddings."""

    def __init__(self, device: torch.device):
        import open_clip

        self.device = device
        # ViT-B-32 + OpenAI weights: practical size (~151M params / ~338MB),
        # strong zero-shot. First run downloads weights into the Hugging Face /
        # open_clip cache; later startups reuse the cache.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model.to(device)
        self.model.eval()

        self._n_plant = len(PLANT_PROMPTS)
        self._all_prompts = PLANT_PROMPTS + NON_PLANT_PROMPTS
        self._text_features: Optional[torch.Tensor] = None
        self._logit_scale: Optional[torch.Tensor] = None
        self._cache_text_embeddings()

        logger.info(
            "Plant gate loaded (OpenCLIP ViT-B-32, device=%s, nonplant_threshold=%.2f)",
            device,
            PLANT_GATE_NONPLANT_THRESHOLD,
        )

    def _cache_text_embeddings(self) -> None:
        """Encode all prompts once at startup and L2-normalize."""
        with torch.inference_mode():
            tokens = self.tokenizer(self._all_prompts).to(self.device)
            text_feat = self.model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            self._text_features = text_feat
            # CLIP temperature scale (cached as a detached tensor).
            self._logit_scale = self.model.logit_scale.exp().detach()

    def classify(self, image: Image.Image) -> PlantGateResult:
        if self._text_features is None or self._logit_scale is None:
            raise RuntimeError("Plant gate text embeddings are not cached")

        # CLIP preprocess is separate from ResNet ImageNet preprocess.
        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            image_feat = self.model.encode_image(image_tensor)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            # Standard CLIP zero-shot: softmax over ALL prompt similarities.
            logits = self._logit_scale * (image_feat @ self._text_features.T)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            plant_score = float(probs[: self._n_plant].sum().item())
            non_plant_score = float(probs[self._n_plant :].sum().item())
            top_idx = int(torch.argmax(probs).item())
            top_is_non_plant = top_idx >= self._n_plant

        # Conservative dual gate — see module comment on PLANT_GATE_NONPLANT_THRESHOLD.
        if (
            non_plant_score >= PLANT_GATE_NONPLANT_THRESHOLD
            and top_is_non_plant
        ):
            decision = "non_plant"
            is_plant = False
        elif plant_score >= PLANT_GATE_NONPLANT_THRESHOLD and not top_is_non_plant:
            decision = "plant"
            is_plant = True
        else:
            decision = "uncertain"
            is_plant = True  # prefer false accepts over false rejects of leaves

        logger.debug(
            "Plant gate: plant_score=%.4f non_plant_score=%.4f "
            "top_is_non_plant=%s decision=%s",
            plant_score,
            non_plant_score,
            top_is_non_plant,
            decision,
        )
        return PlantGateResult(
            is_plant=is_plant,
            plant_score=plant_score,
            non_plant_score=non_plant_score,
            decision=decision,
        )
