"""Text and speech-bubble detection.

Primary path: a YOLO model fine-tuned on comics (ogkalu's detector on
Hugging Face, loaded through ultralytics). Fallback path: a classical
OpenCV pipeline (adaptive threshold + morphology + contour grouping) that
needs no deep-learning dependencies, so the app still works on a bare CPU
install.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import ProcessingConfig
from .utils import TextRegion, resize_longer_side, guess_direction

log = logging.getLogger("manga_translator")

HF_DETECTOR_REPO = "ogkalu/comic-text-and-bubble-detector"

# normalised class names we care about; anything else is ignored
_BUBBLE_NAMES = {"bubble", "speech_bubble", "speech-bubble"}
_TEXT_NAMES = {"text", "text_bubble", "text-bubble", "text_free", "text-free",
               "free_text", "free-text"}


class BaseDetector:
    name = "base"

    def detect(self, img: np.ndarray) -> Tuple[List[TextRegion], List[Tuple[int, int, int, int]]]:
        raise NotImplementedError


class YoloDetector(BaseDetector):
    """ogkalu/comic-text-and-bubble-detector via ultralytics."""

    name = "yolo"

    def __init__(self, cfg: ProcessingConfig):
        from huggingface_hub import list_repo_files, hf_hub_download
        from ultralytics import YOLO

        model_dir = Path(cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        weights = None
        for fname in list_repo_files(HF_DETECTOR_REPO):
            if fname.endswith(".pt"):
                weights = hf_hub_download(HF_DETECTOR_REPO, fname, local_dir=str(model_dir))
                break
        if not weights:
            raise RuntimeError(f"No .pt weights found in {HF_DETECTOR_REPO}")
        self.model = YOLO(weights)
        self.device = cfg.resolved_device()
        self.conf = cfg.box_threshold
        self.det_size = cfg.detection_size

    def detect(self, img: np.ndarray):
        small, scale = resize_longer_side(img, self.det_size)
        results = self.model.predict(small, conf=self.conf, device=self.device, verbose=False)
        regions: List[TextRegion] = []
        bubbles: List[Tuple[int, int, int, int]] = []
        for r in results:
            names = r.names or {}
            for box in r.boxes:
                cls_idx = int(box.cls.item())
                label = str(names.get(cls_idx, cls_idx)).lower()
                x1, y1, x2, y2 = (int(v / scale) for v in box.xyxy[0].tolist())
                conf = float(box.conf.item())
                if label in _BUBBLE_NAMES:
                    bubbles.append((x1, y1, x2, y2))
                else:  # any other class is some flavour of text
                    kind = "text" if label in {"text_free", "free-text", "free_text", "text-free"} else "text"
                    regions.append(TextRegion(x1, y1, x2, y2, kind=kind, confidence=conf))
        _associate_bubbles(regions, bubbles)
        return regions, bubbles


class OpenCVDetector(BaseDetector):
    """Dependency-light fallback detector.

    - Text: adaptive threshold -> morphological closing -> contours grouped
      into blocks. Works well for high-contrast manga lettering.
    - Bubbles: large, bright, roughly convex contours.
    """

    name = "opencv"

    def __init__(self, cfg: ProcessingConfig):
        self.min_area = cfg.min_region_area
        self.det_size = cfg.detection_size

    def detect(self, img: np.ndarray):
        import cv2
        small, scale = resize_longer_side(img, self.det_size)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        # --- text candidates: dark ink on light background ----------------
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)

        # strip long straight lines (panel borders, speed lines) so they
        # don't merge with or swallow text blocks
        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1)))
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80)))
        bw = cv2.subtract(bw, cv2.bitwise_or(horiz, vert))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)), iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: List[TextRegion] = []
        inv = 1.0 / scale
        img_area = gray.shape[0] * gray.shape[1]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area or w * h > img_area * 0.30:
                continue
            if w < 8 or h < 8:
                continue  # line fragments
            if max(w / max(h, 1), h / max(w, 1)) > 14:
                continue  # leftover rules / gutters
            # ink density filter: real text blocks have a decent amount of dark pixels
            density = bw[y:y + h, x:x + w].mean() / 255.0
            if density < 0.03 or density > 0.9:
                continue
            regions.append(TextRegion(int(x * inv), int(y * inv),
                                      int((x + w) * inv), int((y + h) * inv)))
        regions = _merge_boxes(regions)

        # --- speech bubbles: bright blobs ringed by a darker outline ------
        _, bright = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        bcontours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        for c in bcontours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area < img_area * 0.004 or area > img_area * 0.30:
                continue
            hull = cv2.convexHull(c)
            solidity = area / max(cv2.contourArea(hull), 1)
            if solidity < 0.75:
                continue
            # ring-contrast check: a bubble is brighter than its surroundings
            mask_in = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask_in, [c], -1, 255, -1)
            ring = cv2.dilate(mask_in, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (21, 21))) - mask_in
            inside_mean = cv2.mean(gray, mask=mask_in)[0]
            ring_mean = cv2.mean(gray, mask=ring)[0]
            if inside_mean - ring_mean < 12:
                continue
            bubbles.append((int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv)))

        _associate_bubbles(regions, bubbles)
        return regions, bubbles


def _associate_bubbles(regions: List[TextRegion], bubbles: List[Tuple[int, int, int, int]]):
    for reg in regions:
        cx, cy = (reg.x1 + reg.x2) / 2, (reg.y1 + reg.y2) / 2
        for b in bubbles:
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                reg.kind = "bubble-text"
                reg.bubble = b
                break
        reg.direction = guess_direction(reg)


def _merge_boxes(regions: List[TextRegion], iou_thresh: float = 0.2) -> List[TextRegion]:
    """Merge overlapping / contained boxes."""
    merged: List[TextRegion] = []
    for reg in sorted(regions, key=lambda r: r.area, reverse=True):
        placed = False
        for m in merged:
            ix1, iy1 = max(reg.x1, m.x1), max(reg.y1, m.y1)
            ix2, iy2 = min(reg.x2, m.x2), min(reg.y2, m.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter and inter / min(reg.area, m.area) > iou_thresh:
                m.x1, m.y1 = min(m.x1, reg.x1), min(m.y1, reg.y1)
                m.x2, m.y2 = max(m.x2, reg.x2), max(m.y2, reg.y2)
                placed = True
                break
        if not placed:
            merged.append(reg)
    merged.sort(key=lambda r: (r.y1, r.x1))
    return merged


_detector_cache: dict = {}


def get_detector(cfg: ProcessingConfig) -> BaseDetector:
    """Factory with graceful degradation yolo -> opencv."""
    choice = cfg.detector
    if choice in ("auto", "yolo"):
        try:
            if "yolo" not in _detector_cache:
                _detector_cache["yolo"] = YoloDetector(cfg)
            return _detector_cache["yolo"]
        except Exception as e:
            if choice == "yolo":
                raise
            log.warning("YOLO detector unavailable (%s); using OpenCV fallback.", e)
    if "opencv" not in _detector_cache:
        _detector_cache["opencv"] = OpenCVDetector(cfg)
    return _detector_cache["opencv"]
