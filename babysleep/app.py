"""Baby prone-sleep monitor web app.

Reads a camera / RTSP stream / video file, runs YOLO26, and serves a web page
(viewable from a phone on the same Wi-Fi) with the live video, the current
posture and an audible alarm when the baby has been prone for too long.

    python -m babysleep.app --source 0
    python -m babysleep.app --source rtsp://user:pass@192.168.1.20/stream1
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify

from .detector import PostureDetector
from .monitor import MonitorConfig, MonitorState, SleepMonitor
from .posture import Posture, PostureConfig

log = logging.getLogger("babysleep")

_COLORS = {  # BGR
    Posture.SUPINE: (80, 200, 80),
    Posture.SIDE: (0, 200, 255),
    Posture.PRONE: (0, 0, 255),
    Posture.UNKNOWN: (160, 160, 160),
}


class Notifier:
    """POSTs a JSON message to a webhook (e.g. ntfy.sh, Discord, Home Assistant)."""

    def __init__(self, url: str | None, cooldown: float = 60.0):
        self.url = url
        self.cooldown = cooldown
        self._last = 0.0

    def notify(self, message: str) -> None:
        now = time.time()
        if not self.url or now - self._last < self.cooldown:
            return
        self._last = now
        threading.Thread(target=self._post, args=(message,), daemon=True).start()

    def _post(self, message: str) -> None:
        try:
            requests.post(self.url, json={"text": message, "content": message}, timeout=5)
        except requests.RequestException as e:
            log.warning("webhook failed: %s", e)


class MonitorWorker(threading.Thread):
    def __init__(self, source: str, detector: PostureDetector, monitor: SleepMonitor, notifier: Notifier):
        super().__init__(daemon=True)
        self.source = int(source) if source.isdigit() else source
        self.detector = detector
        self.monitor = monitor
        self.notifier = notifier
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status: dict = {"posture": Posture.UNKNOWN.value, "alert": False, "running": False}

    def run(self) -> None:
        while True:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                log.error("cannot open source %s, retrying in 5s", self.source)
                time.sleep(5)
                continue
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self._process(frame)
            # Stream dropped or video file ended: reconnect (video files loop).
            cap.release()
            time.sleep(0.5)

    def _process(self, frame: np.ndarray) -> None:
        det = self.detector(frame)
        state = self.monitor.update(det.result.posture, time.monotonic())
        if state.alert:
            self.notifier.notify(f"⚠️ 寶寶趴睡已 {state.prone_seconds:.0f} 秒，請立即查看！")
        elif state.missing:
            self.notifier.notify("⚠️ 畫面中偵測不到寶寶，請查看。")

        img = _draw_overlay(det.annotated, state)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self.lock:
            if ok:
                self.jpeg = buf.tobytes()
            self.status = {
                "running": True,
                "posture": state.posture.value,
                "posture_zh": state.posture.zh,
                "frame_posture": det.result.posture.value,
                "confidence": round(det.result.confidence, 3),
                "reason": det.result.reason,
                "prone_seconds": round(state.prone_seconds, 1),
                "alert": state.alert,
                "missing": state.missing,
                "time": time.time(),
            }


def _draw_overlay(img: np.ndarray, state: MonitorState) -> np.ndarray:
    color = (0, 0, 255) if state.alert else _COLORS[state.posture]
    label = f"{state.posture.value.upper()}"
    if state.posture is Posture.PRONE:
        label += f" {state.prone_seconds:.0f}s"
    if state.alert:
        label = "ALERT! " + label
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 12)
    cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    return img


_INDEX_HTML = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>寶寶睡姿監測</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee;text-align:center}
#status{font-size:2rem;padding:16px;font-weight:bold}
.supine{background:#2e7d32}.side{background:#f9a825;color:#111}
.prone{background:#e65100}.unknown{background:#555}.alert{background:#c62828;animation:b 1s infinite}
@keyframes b{50%{opacity:.5}}
img{max-width:100%;height:auto}
button{font-size:1rem;padding:10px 20px;margin:12px}
small{color:#aaa}
</style></head><body>
<div id="status" class="unknown">連線中…</div>
<img src="/stream" alt="video">
<div><button id="arm">🔔 啟用警報聲音</button></div>
<small id="detail"></small>
<script>
let ctx=null;
document.getElementById('arm').onclick=()=>{
  ctx=new (window.AudioContext||window.webkitAudioContext)();
  document.getElementById('arm').textContent='🔔 警報聲音已啟用';
  if('Notification' in window) Notification.requestPermission();
};
function beep(){ if(!ctx) return;
  const o=ctx.createOscillator(),g=ctx.createGain();
  o.frequency.value=880;o.connect(g);g.connect(ctx.destination);
  g.gain.value=0.3;o.start();o.stop(ctx.currentTime+0.4);
}
let notified=false;
async function poll(){
  try{
    const s=await (await fetch('/api/status')).json();
    const el=document.getElementById('status');
    let text='睡姿：'+(s.posture_zh||'未知');
    if(s.posture==='prone') text+='（'+s.prone_seconds+' 秒）';
    if(s.alert) text='⚠️ 寶寶趴睡！請立即查看';
    else if(s.missing) text='⚠️ 偵測不到寶寶';
    el.textContent=text;
    el.className=s.alert?'alert':(s.posture||'unknown');
    document.getElementById('detail').textContent=s.reason?('單幀：'+s.frame_posture+' / '+s.reason):'';
    if(s.alert){ beep(); navigator.vibrate&&navigator.vibrate(300);
      if(!notified&&'Notification' in window&&Notification.permission==='granted'){new Notification('寶寶趴睡警報');notified=true;}
    } else notified=false;
  }catch(e){ document.getElementById('status').textContent='連線中斷'; }
  setTimeout(poll,1000);
}
poll();
</script></body></html>"""


def create_app(worker: MonitorWorker) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return _INDEX_HTML

    @app.get("/api/status")
    def status():
        with worker.lock:
            return jsonify(worker.status)

    @app.get("/stream")
    def stream():
        def gen():
            last = None
            while True:
                with worker.lock:
                    jpeg = worker.jpeg
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.05)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="YOLO26 嬰兒趴睡偵測")
    p.add_argument("--source", default="0", help="攝影機編號、RTSP/HTTP 網址或影片檔")
    p.add_argument("--model", default="yolo26n-pose.pt", help="YOLO26 pose 模型，或 train.py 訓練出的自訂模型")
    p.add_argument("--conf", type=float, default=0.25, help="偵測信心門檻")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu、0（GPU）、mps …")
    p.add_argument("--kpt-conf", type=float, default=0.5, help="關鍵點可見門檻")
    p.add_argument("--alert-after", type=float, default=5.0, help="持續趴睡幾秒後警報")
    p.add_argument("--missing-after", type=float, default=30.0, help="幾秒沒看到寶寶就提醒（0 停用）")
    p.add_argument("--webhook", default=None, help="警報時 POST 的 webhook 網址（如 ntfy.sh）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detector = PostureDetector(args.model, args.conf, args.imgsz, args.device, PostureConfig(kpt_conf=args.kpt_conf))
    monitor = SleepMonitor(MonitorConfig(alert_after_seconds=args.alert_after, missing_after_seconds=args.missing_after))
    worker = MonitorWorker(args.source, detector, monitor, Notifier(args.webhook))
    worker.start()

    log.info("open http://<this-computer-ip>:%d on your phone", args.port)
    create_app(worker).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
