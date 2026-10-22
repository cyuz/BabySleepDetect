"""Sleep-posture rules based on COCO-17 keypoints from a YOLO26 pose model.

This module is pure numpy so it can be unit-tested without a model.

COCO keypoint order:
    0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
    5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
    9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


class Posture(str, Enum):
    SUPINE = "supine"  # 仰睡
    PRONE = "prone"  # 趴睡
    SIDE = "side"  # 側睡
    UNKNOWN = "unknown"  # 無法判斷 / 未偵測到嬰兒

    @property
    def zh(self) -> str:
        return {"supine": "仰睡", "prone": "趴睡", "side": "側睡", "unknown": "未知"}[self.value]


@dataclass
class PostureResult:
    posture: Posture
    confidence: float
    reason: str


@dataclass
class PostureConfig:
    kpt_conf: float = 0.5  # a keypoint counts as visible above this confidence
    side_ratio: float = 0.35  # shoulder width / torso length below this => side-lying


def _visible(kpts: np.ndarray, idx: int, thr: float) -> bool:
    return bool(kpts[idx, 2] >= thr)


def _mid(kpts: np.ndarray, a: int, b: int, thr: float) -> np.ndarray | None:
    pts = [kpts[i, :2] for i in (a, b) if _visible(kpts, i, thr)]
    return np.mean(pts, axis=0) if pts else None


def classify_posture(kpts: np.ndarray, cfg: PostureConfig | None = None) -> PostureResult:
    """Classify one person's sleep posture from a (17, 3) array of (x, y, conf)."""
    cfg = cfg or PostureConfig()
    kpts = np.asarray(kpts, dtype=float)
    thr = cfg.kpt_conf

    face_conf = float(np.mean(kpts[[NOSE, L_EYE, R_EYE], 2]))
    face_visible = face_conf >= thr
    both_shoulders = _visible(kpts, L_SHOULDER, thr) and _visible(kpts, R_SHOULDER, thr)
    shoulder_mid = _mid(kpts, L_SHOULDER, R_SHOULDER, thr)
    hip_mid = _mid(kpts, L_HIP, R_HIP, thr)
    torso_visible = shoulder_mid is not None or hip_mid is not None

    if not torso_visible and not face_visible:
        return PostureResult(Posture.UNKNOWN, 0.0, "body not visible")

    # Facing direction from the body's own frame: with "up" = hips -> shoulders,
    # a body facing the camera has its left shoulder on the right of that axis.
    facing = 0.0  # >0 facing camera, <0 back to camera
    if both_shoulders and hip_mid is not None:
        up = shoulder_mid - hip_mid
        lr = kpts[L_SHOULDER, :2] - kpts[R_SHOULDER, :2]
        torso_len = float(np.linalg.norm(up))
        shoulder_w = float(np.linalg.norm(lr))
        if torso_len > 1e-6:
            if shoulder_w / torso_len < cfg.side_ratio:
                return PostureResult(Posture.SIDE, 0.6, "narrow shoulders relative to torso")
            cross = up[0] * lr[1] - up[1] * lr[0]
            facing = float(np.clip(cross / (torso_len * max(shoulder_w, 1e-6)), -1, 1))

    if face_visible:
        if facing < -0.5:
            # Face keypoints seen while the torso faces away: head turned while prone.
            return PostureResult(Posture.PRONE, 0.6, "torso facing away, head turned")
        return PostureResult(Posture.SUPINE, face_conf, "face visible")

    # Face not visible but body is: face-down or face covered — both are unsafe.
    if facing < -0.3:
        return PostureResult(Posture.PRONE, 0.9, "face hidden, back facing camera")
    return PostureResult(Posture.PRONE, 0.7, "face hidden or covered")
