"""YOLO26 inference wrappers that turn a frame into a sleep posture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .posture import Posture, PostureConfig, PostureResult, classify_posture

# Class-name keywords accepted from a custom-trained model.
_NAME_TO_POSTURE = {
    "prone": Posture.PRONE,
    "stomach": Posture.PRONE,
    "supine": Posture.SUPINE,
    "back": Posture.SUPINE,
    "side": Posture.SIDE,
}


def posture_from_class_name(name: str) -> Posture:
    name = name.lower()
    for key, posture in _NAME_TO_POSTURE.items():
        if key in name:
            return posture
    return Posture.UNKNOWN


@dataclass
class Detection:
    result: PostureResult
    box: np.ndarray | None  # xyxy of the tracked baby
    annotated: np.ndarray  # frame with model drawings


class PostureDetector:
    """Detects the baby's posture with a YOLO26 model.

    * Pose model (default ``yolo26n-pose.pt``): rule-based posture from keypoints,
      works out of the box.
    * Custom model (``yolo26*.pt`` trained by ``train.py`` on prone/supine/side
      classes, detect or classify task): used directly when provided.
    """

    def __init__(
        self,
        model: str = "yolo26n-pose.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
        posture_cfg: PostureConfig | None = None,
    ):
        from ultralytics import YOLO  # imported lazily so posture logic stays light

        self.model = YOLO(model)
        self.task = self.model.task
        if self.task not in ("pose", "detect", "classify"):
            raise ValueError(f"Unsupported YOLO task '{self.task}', use a pose/detect/classify model")
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.posture_cfg = posture_cfg or PostureConfig()

    def __call__(self, frame: np.ndarray) -> Detection:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        annotated = r.plot()

        if self.task == "classify":
            top = int(r.probs.top1)
            posture = posture_from_class_name(r.names[top])
            return Detection(PostureResult(posture, float(r.probs.top1conf), f"class '{r.names[top]}'"), None, annotated)

        if r.boxes is None or len(r.boxes) == 0:
            return Detection(PostureResult(Posture.UNKNOWN, 0.0, "no baby detected"), None, annotated)

        # Assume the baby is the largest subject in a crib camera view.
        xyxy = r.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        i = int(np.argmax(areas))

        if self.task == "pose":
            kpts = r.keypoints.data[i].cpu().numpy()
            result = classify_posture(kpts, self.posture_cfg)
        else:
            cls = int(r.boxes.cls[i])
            result = PostureResult(posture_from_class_name(r.names[cls]), float(r.boxes.conf[i]), f"class '{r.names[cls]}'")
        return Detection(result, xyxy[i], annotated)
