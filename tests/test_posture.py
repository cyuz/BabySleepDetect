import numpy as np
import pytest

from babysleep.detector import posture_from_class_name
from babysleep.monitor import MonitorConfig, SleepMonitor
from babysleep.posture import Posture, classify_posture


def make_kpts(face=True, facing_camera=True, shoulder_w=60.0, rotate_deg=0.0):
    """Synthetic COCO-17 keypoints of a body lying head-up in the image."""
    k = np.zeros((17, 3))
    half = shoulder_w / 2
    # From the camera, a body facing it has its left shoulder on the image right.
    sign = 1 if facing_camera else -1
    k[5] = [100 + sign * half, 100, 0.9]  # left shoulder
    k[6] = [100 - sign * half, 100, 0.9]  # right shoulder
    k[11] = [100 + sign * 20, 200, 0.9]  # left hip
    k[12] = [100 - sign * 20, 200, 0.9]  # right hip
    k[3] = [100 + sign * 15, 60, 0.8]  # left ear
    k[4] = [100 - sign * 15, 60, 0.8]  # right ear
    if face:
        k[0] = [100, 55, 0.9]
        k[1] = [110, 45, 0.9]
        k[2] = [90, 45, 0.9]
    if rotate_deg:
        a = np.deg2rad(rotate_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        k[:, :2] = (k[:, :2] - 100) @ rot.T + 100
    return k


@pytest.mark.parametrize("rotate", [0, 90, 180, 270, 45])
def test_supine_any_orientation(rotate):
    assert classify_posture(make_kpts(rotate_deg=rotate)).posture is Posture.SUPINE


@pytest.mark.parametrize("rotate", [0, 90, 180, 270, 45])
def test_prone_back_to_camera(rotate):
    r = classify_posture(make_kpts(face=False, facing_camera=False, rotate_deg=rotate))
    assert r.posture is Posture.PRONE
    assert r.confidence >= 0.9


def test_prone_head_turned_face_visible():
    assert classify_posture(make_kpts(face=True, facing_camera=False)).posture is Posture.PRONE


def test_face_covered_is_flagged_prone():
    assert classify_posture(make_kpts(face=False, facing_camera=True)).posture is Posture.PRONE


def test_side_lying():
    assert classify_posture(make_kpts(shoulder_w=10)).posture is Posture.SIDE


def test_nothing_visible():
    assert classify_posture(np.zeros((17, 3))).posture is Posture.UNKNOWN


def test_class_name_mapping():
    assert posture_from_class_name("baby_prone") is Posture.PRONE
    assert posture_from_class_name("Supine") is Posture.SUPINE
    assert posture_from_class_name("side") is Posture.SIDE
    assert posture_from_class_name("person") is Posture.UNKNOWN


def test_monitor_alerts_after_persistent_prone():
    m = SleepMonitor(MonitorConfig(window_seconds=2, alert_after_seconds=5))
    t = 0.0
    for _ in range(10):  # 1s of supine
        assert not m.update(Posture.SUPINE, t).alert
        t += 0.1
    states = []
    for _ in range(80):  # 8s of prone
        states.append(m.update(Posture.PRONE, t))
        t += 0.1
    assert not states[30].alert
    assert states[-1].alert
    assert states[-1].posture is Posture.PRONE


def test_monitor_ignores_brief_prone_flicker():
    m = SleepMonitor(MonitorConfig(window_seconds=2, alert_after_seconds=5))
    t = 0.0
    for i in range(200):
        s = m.update(Posture.PRONE if i % 5 == 0 else Posture.SUPINE, t)
        assert not s.alert
        t += 0.1


def test_monitor_missing_baby():
    m = SleepMonitor(MonitorConfig(missing_after_seconds=10))
    m.update(Posture.SUPINE, 0.0)
    assert not m.update(Posture.UNKNOWN, 5.0).missing
    assert m.update(Posture.UNKNOWN, 11.0).missing
