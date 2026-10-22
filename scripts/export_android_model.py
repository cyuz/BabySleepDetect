"""Export a YOLO26 model to ONNX for the Android app.

    python scripts/export_android_model.py                          # default pose model
    python scripts/export_android_model.py --model runs/detect/baby_sleep/weights/best.pt

Writes android/app/src/main/assets/model.onnx. The app reads task / class names
from the ONNX metadata, so pose, detect and classify models all work.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ASSET = Path(__file__).resolve().parent.parent / "android/app/src/main/assets/model.onnx"


def main() -> None:
    p = argparse.ArgumentParser(description="匯出 YOLO26 ONNX 模型給 Android App")
    p.add_argument("--model", default="yolo26n-pose.pt", help="YOLO26 pose 模型或 train.py 訓練出的模型")
    p.add_argument("--imgsz", type=int, default=640, help="輸入尺寸，手機較慢可用 480 或 320")
    args = p.parse_args()

    model = YOLO(args.model)
    kwargs = {} if model.task == "classify" else {"end2end": True}  # NMS-free output for detect / pose
    exported = model.export(format="onnx", imgsz=args.imgsz, simplify=True, **kwargs)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, ASSET)
    print(f"Saved {ASSET}")


if __name__ == "__main__":
    main()
