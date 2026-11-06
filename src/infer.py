"""ONNX Runtime classifier – the shared Classifier interface used by all consumers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class Classifier:
    """Load an ONNX model and classify BGR images.

    The exported graph already contains ImageNet normalisation, so this class
    only needs to:
      1. Convert BGR → RGB
      2. Resize to 224×224
      3. Scale uint8 → float32 in [0, 1]
      4. Add a batch dimension and run ORT

    Parameters
    ----------
    onnx_path:
        Path to ``models/model.onnx``.
    labels_path:
        Path to ``models/labels.json`` – an ordered list of class names.
    """

    def __init__(self, onnx_path: str | Path, labels_path: str | Path) -> None:
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name

        with open(labels_path, encoding="utf-8") as fh:
            self._labels: list[str] = json.load(fh)

    # ------------------------------------------------------------------
    # Public interface (do not change without updating shared interfaces)
    # ------------------------------------------------------------------

    def predict(
        self, image_bgr: np.ndarray
    ) -> tuple[str, float, dict[str, float]]:
        """Classify one BGR image.

        Parameters
        ----------
        image_bgr:
            OpenCV BGR image, any size, uint8 or float.

        Returns
        -------
        label:
            Predicted class name.
        confidence:
            Softmax probability of the top class (0–1).
        probs:
            Dict mapping every class name to its softmax probability.
        """
        tensor = self._preprocess(image_bgr)
        logits: np.ndarray = self._session.run(None, {self._input_name: tensor})[0][0]
        probs = _softmax(logits)

        top_idx = int(probs.argmax())
        label = self._labels[top_idx]
        confidence = float(probs[top_idx])
        prob_dict = {name: float(probs[i]) for i, name in enumerate(self._labels)}
        return label, confidence, prob_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → NCHW float32 [0, 1] RGB, 224×224."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0          # [0, 1]
        tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
        return np.ascontiguousarray(tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
