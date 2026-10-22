"""Fine-tune YOLO26 on a baby sleep-posture dataset (supine / prone / side).

Detection (bounding boxes, see data/baby_sleep.yaml):
    python train.py --data data/baby_sleep.yaml --model yolo26n.pt

Classification (folder per class: datasets/baby_cls/{train,val}/{supine,prone,side}/*.jpg):
    python train.py --data datasets/baby_cls --model yolo26n-cls.pt

Then run the monitor with the trained weights:
    python -m babysleep.app --model runs/detect/baby_sleep/weights/best.pt
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description="訓練 YOLO26 嬰兒睡姿模型")
    p.add_argument("--data", default="data/baby_sleep.yaml", help="偵測用 dataset yaml，或分類用資料夾")
    p.add_argument("--model", default="yolo26n.pt", help="yolo26n.pt / yolo26s.pt / yolo26n-cls.pt …")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default="baby_sleep")
    p.add_argument("--export", default=None, help="訓練完匯出格式，如 onnx、tflite、coreml、ncnn")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        fliplr=0.5,
        degrees=180,  # babies lie in any orientation when seen from above
    )
    model.val()
    if args.export:
        model.export(format=args.export, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
