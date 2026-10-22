"""Temporal smoothing and alert logic on top of per-frame posture results."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .posture import Posture


@dataclass
class MonitorConfig:
    window_seconds: float = 3.0  # smoothing window for the per-frame votes
    alert_after_seconds: float = 5.0  # prone must persist this long before alarming
    missing_after_seconds: float = 30.0  # warn if no baby seen this long (0 disables)
    prone_ratio: float = 0.6  # fraction of prone frames in the window to count as prone


@dataclass
class MonitorState:
    posture: Posture
    alert: bool
    prone_seconds: float
    missing: bool


class SleepMonitor:
    def __init__(self, cfg: MonitorConfig | None = None):
        self.cfg = cfg or MonitorConfig()
        self._votes: deque[tuple[float, Posture]] = deque()
        self._prone_since: float | None = None
        self._last_seen: float | None = None

    def update(self, posture: Posture, t: float) -> MonitorState:
        cfg = self.cfg
        if self._last_seen is None:
            self._last_seen = t
        if posture is not Posture.UNKNOWN:
            self._last_seen = t

        self._votes.append((t, posture))
        while self._votes and t - self._votes[0][0] > cfg.window_seconds:
            self._votes.popleft()

        smoothed = self._smoothed()
        if smoothed is Posture.PRONE:
            if self._prone_since is None:
                self._prone_since = t
        else:
            self._prone_since = None

        prone_seconds = 0.0 if self._prone_since is None else t - self._prone_since
        missing = cfg.missing_after_seconds > 0 and t - self._last_seen >= cfg.missing_after_seconds
        return MonitorState(
            posture=smoothed,
            alert=prone_seconds >= cfg.alert_after_seconds,
            prone_seconds=prone_seconds,
            missing=missing,
        )

    def _smoothed(self) -> Posture:
        known = [p for _, p in self._votes if p is not Posture.UNKNOWN]
        if not known:
            return Posture.UNKNOWN
        if known.count(Posture.PRONE) / len(known) >= self.cfg.prone_ratio:
            return Posture.PRONE
        return max(set(known), key=known.count)
