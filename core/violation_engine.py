"""
Violation detection engine — CBVMS.

This module implements lightweight computer-vision heuristics (hair + ID badge)
and optional MobileNetV2 classifiers (uniform + earring) when model files exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Easy-to-edit HSV constants for "allowed dark hair" policy.
# Tune these if you have false positives/negatives.
HAIR_ALLOWED_HSV_LOWER = (0, 0, 0)  # H, S, V
HAIR_ALLOWED_HSV_UPPER = (180, 140, 120)  # H, S, V


class ViolationEngine:
    """
    Evaluates a detected person bounding box for policy violations.

    check_all() returns a list of violation strings (e.g. ["wrong_hair_color"]).
    """

    # Class attributes (as requested) to make policy tuning simple.
    ALLOWED_HSV_LOWER = HAIR_ALLOWED_HSV_LOWER
    ALLOWED_HSV_UPPER = HAIR_ALLOWED_HSV_UPPER

    HAIR_OUTSIDE_THRESHOLD_FRAC = 0.15  # >15% outside allowed => violation

    def __init__(self) -> None:
        self._device: str = "cpu"
        self._torch: Any | None = None
        self._torchvision: Any | None = None

        self._uniform_model = None
        self._earring_model = None
        self._uniform_transform = None
        self._earring_transform = None

        # Output labels used by your checkpoints.
        # (If your checkpoint uses a different label ordering, update these.)
        self._uniform_class_labels = ["correct_uniform", "wrong_uniform"]
        self._earring_class_labels = ["without_earring", "with_earring"]

        self._load_optional_classifiers()

    def _load_optional_classifiers(self) -> None:
        # Model files are expected at: <project_root>/models/*.pth
        project_root = Path(__file__).resolve().parents[2]
        models_dir = project_root / "models"

        uniform_path = models_dir / "uniform_model.pth"
        earring_path = models_dir / "earring_model.pth"

        try:
            import torch  # type: ignore
            from torchvision import models as tv_models  # type: ignore
            from torchvision import transforms as tv_transforms  # type: ignore
            from PIL import Image  # noqa: F401  # type: ignore
        except Exception:
            print("[CBVMS] Warning: torch/torchvision not available; skipping uniform/earring checks.")
            return

        self._torch = torch
        self._torchvision = tv_models
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        common_transform = tv_transforms.Compose(
            [
                tv_transforms.Resize((224, 224)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._uniform_transform = common_transform
        self._earring_transform = common_transform

        if uniform_path.exists():
            self._uniform_model = self._load_mobilenet(
                checkpoint_path=uniform_path,
                num_classes=len(self._uniform_class_labels),
            )
        else:
            print("[CBVMS] Warning: uniform_model.pth not found; skipping uniform checks.")

        if earring_path.exists():
            self._earring_model = self._load_mobilenet(
                checkpoint_path=earring_path,
                num_classes=len(self._earring_class_labels),
            )
        else:
            print("[CBVMS] Warning: earring_model.pth not found; skipping earring checks.")

    def _load_mobilenet(self, checkpoint_path: Path, num_classes: int):
        assert self._torch is not None
        torch = self._torch
        tv_models = self._torchvision
        assert tv_models is not None

        model = tv_models.mobilenet_v2(weights=None)
        # Replace final classifier head for the expected number of classes.
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            state_dict = None

            if isinstance(checkpoint, dict):
                for key in ("state_dict", "model_state_dict", "net_state_dict", "model"):
                    maybe = checkpoint.get(key)
                    if isinstance(maybe, dict):
                        state_dict = maybe
                        break

                # Sometimes checkpoints are already a raw state_dict.
                if state_dict is None and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint

            if state_dict is None:
                print(f"[CBVMS] Warning: {checkpoint_path.name} has unexpected format; skipping load.")
                return None

            # Handle DataParallel("module.") prefix.
            normalized = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    normalized[k.replace("module.", "", 1)] = v
                else:
                    normalized[k] = v

            model.load_state_dict(normalized, strict=False)
        except Exception as exc:
            print(f"[CBVMS] Warning: failed loading {checkpoint_path.name}: {exc}")
            return None

        model.to(self._device)
        model.eval()
        return model

    def detect_gender(self, person_crop: np.ndarray) -> str:
        """
        Detect gender using DeepFace (optional).

        Returns: "Man", "Woman", or "unknown"
        """

        try:
            from deepface import DeepFace  # type: ignore
        except Exception:
            return "unknown"

        try:
            if person_crop is None or person_crop.size == 0:
                return "unknown"

            # DeepFace expects RGB arrays.
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            analysis = DeepFace.analyze(
                img=rgb,
                enforce_detection=False,
                silent=True,
            )
            if isinstance(analysis, list):
                analysis = analysis[0] if analysis else {}

            dominant = str(analysis.get("dominant_gender", analysis.get("gender", "")) or "").strip()
            if not dominant:
                return "unknown"

            d = dominant.lower()
            if d.startswith("m") or "male" in d:
                return "Man"
            if d.startswith("f") or "female" in d or "woman" in d:
                return "Woman"
            # Some models return "Man"/"Woman" directly.
            if dominant in ("Man", "Woman"):
                return dominant
            return "unknown"
        except Exception:
            return "unknown"

    def check_all(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        gender: str = "unknown",
    ) -> list[str]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return []

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return []

        violations: list[str] = []
        if self.check_hair_color(person_crop):
            violations.append("wrong_hair_color")
        if self.check_id_badge(person_crop):
            violations.append("no_id_badge")
        if self.check_uniform(person_crop):
            violations.append("wrong_uniform")
        if self.check_earring(person_crop, gender):
            violations.append("with_earring")
        return violations

    def check_hair_color(self, person_crop: np.ndarray) -> bool:
        """
        Hair policy: more than 15% of head pixels outside allowed dark HSV range => violation.
        """
        h, w = person_crop.shape[:2]
        head_h = max(1, int(h * 0.30))
        head = person_crop[0:head_h, :]
        if head.size == 0:
            return False

        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
        # Non-allowed mask (built explicitly with cv2.inRange):
        # - Non-allowed if saturation is too high OR value/brightness is too high.
        s_threshold = int(self.ALLOWED_HSV_UPPER[1]) + 1
        v_threshold = int(self.ALLOWED_HSV_UPPER[2]) + 1

        mask_high_s = cv2.inRange(hsv, (0, s_threshold, 0), (180, 255, 255))
        mask_high_v = cv2.inRange(hsv, (0, 0, v_threshold), (180, 255, 255))
        mask_non_allowed = cv2.bitwise_or(mask_high_s, mask_high_v)

        total = head.shape[0] * head.shape[1]
        outside = int(cv2.countNonZero(mask_non_allowed))
        outside_frac = outside / float(total) if total else 0.0
        return outside_frac > self.HAIR_OUTSIDE_THRESHOLD_FRAC

    def check_id_badge(self, person_crop: np.ndarray) -> bool:
        """
        Badge policy (heuristic):
        - Crop chest region (middle 30% vertically, center 60% horizontally)
        - Brightness + rectangular contour check
        - If a small rectangular object is detected => ID present => return False
        - Otherwise => return True (no ID badge)
        """

        h, w = person_crop.shape[:2]
        if h < 10 or w < 10:
            return True

        cy1 = int(h * 0.35)
        cy2 = int(h * 0.65)
        cx1 = int(w * 0.20)
        cx2 = int(w * 0.80)
        cy1, cy2 = max(0, cy1), max(cy1 + 1, cy2)
        cx1, cx2 = max(0, cx1), max(cx1 + 1, cx2)

        chest = person_crop[cy1:cy2, cx1:cx2]
        if chest.size == 0:
            return True

        gray = cv2.cvtColor(chest, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Brightness check: badge-like regions tend to be bright.
        thresh_val = int(max(30, gray.max() * 0.60))
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # Strengthen rectangle edges.
        kernel = np.ones((3, 3), dtype=np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chest_area = float(chest.shape[0] * chest.shape[1])

        badge_found = False
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < chest_area * 0.01 or area > chest_area * 0.25:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue

            aspect = bw / float(bh)
            # Typical badge-ish rectangle: width > height, moderately wide.
            if 1.2 <= aspect <= 4.0 and bh >= 6:
                badge_found = True
                break

        # TODO: Replace this heuristic with a YOLOv8 fine-tuned `id_badge` detector.
        return not badge_found

    def check_uniform(self, person_crop: np.ndarray) -> bool:
        """
        Uniform classifier check (optional).
        Returns True if predicted class is "wrong_uniform".
        """
        if self._uniform_model is None or self._uniform_transform is None or self._torch is None:
            return False
        if person_crop is None or person_crop.size == 0:
            return False

        import PIL.Image  # type: ignore

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)
        x = self._uniform_transform(pil_img).unsqueeze(0).to(self._device)

        with self._torch.inference_mode():
            logits = self._uniform_model(x)
            pred_idx = int(self._torch.argmax(logits, dim=1).item())

        pred_label = (
            self._uniform_class_labels[pred_idx]
            if 0 <= pred_idx < len(self._uniform_class_labels)
            else str(pred_idx)
        )
        return pred_label == "wrong_uniform"

    def check_earring(self, person_crop: np.ndarray, gender: str) -> bool:
        """
        Earring classifier check (optional).
        Applies only if gender == "Man".
        Returns True if predicted class is "with_earring".
        """
        if gender != "Man":
            return False
        if self._earring_model is None or self._earring_transform is None or self._torch is None:
            return False
        if person_crop is None or person_crop.size == 0:
            return False

        import PIL.Image  # type: ignore

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)
        x = self._earring_transform(pil_img).unsqueeze(0).to(self._device)

        with self._torch.inference_mode():
            logits = self._earring_model(x)
            pred_idx = int(self._torch.argmax(logits, dim=1).item())

        pred_label = (
            self._earring_class_labels[pred_idx]
            if 0 <= pred_idx < len(self._earring_class_labels)
            else str(pred_idx)
        )
        return pred_label == "with_earring"


# Backwards-compatible alias for any older imports.
CBVMSViolationEngine = ViolationEngine
