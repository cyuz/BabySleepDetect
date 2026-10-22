# BabySleepDetect — YOLO26 嬰兒趴睡偵測

用 [Ultralytics YOLO26](https://docs.ultralytics.com/) 即時分析嬰兒床攝影機的畫面，判斷寶寶是 **仰睡 / 側睡 / 趴睡**。
趴睡持續超過設定的秒數時，手機網頁會響鈴、震動並跳出通知，也可以推播到 webhook。

> ⚠️ 本專案只是輔助工具，**不能取代大人照顧與醫療級的監測設備**。請遵循安全睡眠建議（仰睡、床上不放軟物）。

## 運作方式

| 模式 | 模型 | 說明 |
|---|---|---|
| 預設（免訓練） | `yolo26n-pose.pt` | 用 YOLO26 姿態估計取得 17 個關鍵點，再依規則判斷睡姿 |
| 自訂模型（較準） | 用 `train.py` 訓練的 YOLO26 detect / classify 模型 | 直接輸出 `supine` / `prone` / `side` 類別 |

姿態規則（`babysleep/posture.py`）：

- **看得到臉**（鼻子、雙眼）→ 仰睡；但如果軀幹背對鏡頭（臉側轉）→ 趴睡
- **看得到身體但看不到臉** → 趴睡（臉朝下或被被子蓋住，兩者都危險）
- 由「臀部 → 肩膀」的方向與左右肩的相對位置判斷正面或背面，**與寶寶在畫面中的方向無關**
- 肩寬相對於軀幹長度很窄 → 側睡

`babysleep/monitor.py` 會在幾秒的時間窗內做多數決平滑，避免單幀誤判。趴睡持續 `--alert-after` 秒才會警報；
畫面中太久偵測不到寶寶時也會提醒。

## 安裝

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

第一次執行時會自動下載 `yolo26n-pose.pt`。

## 使用

```bash
# USB / 筆電攝影機
python -m babysleep.app --source 0

# 網路攝影機（RTSP）
python -m babysleep.app --source "rtsp://user:pass@192.168.1.20:554/stream1"

# 用影片檔測試
python -m babysleep.app --source sample.mp4
```

用連到同一個 Wi-Fi 的手機打開 `http://<電腦IP>:8000`，再按「啟用警報聲音」（瀏覽器要求使用者點擊後才能播放聲音）。

常用參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--model` | `yolo26n-pose.pt` | 可換成 `yolo26s-pose.pt`（較準）或自訂模型 |
| `--alert-after` | `5` | 趴睡持續幾秒才警報 |
| `--missing-after` | `30` | 幾秒偵測不到寶寶就提醒，`0` 為停用 |
| `--kpt-conf` | `0.5` | 關鍵點可見門檻，夜視畫面可調低 |
| `--webhook` | 無 | 警報時 POST `{"text": ..., "content": ...}`，例如 ntfy、Discord webhook |
| `--device` | 自動 | `cpu`、`0`（NVIDIA GPU）、`mps`（Apple） |

API：`GET /api/status` 回傳目前睡姿的 JSON，`GET /stream` 是 MJPEG 影像串流。

## 訓練自訂 YOLO26 模型（建議）

通用姿態模型是用成人資料訓練的，遇到包巾、棉被或夜視畫面時可能不穩定。收集自家嬰兒床的照片並標註後再微調，準確度會高很多：

```bash
# 物件偵測：依 data/baby_sleep.yaml 準備 YOLO 格式資料（類別 supine / prone / side）
python train.py --data data/baby_sleep.yaml --model yolo26n.pt --epochs 100

# 或影像分類：datasets/baby_cls/{train,val}/{supine,prone,side}/*.jpg
python train.py --data datasets/baby_cls --model yolo26n-cls.pt

# 使用訓練結果
python -m babysleep.app --model runs/detect/baby_sleep/weights/best.pt
```

類別名稱只要包含 `prone` / `supine` / `side` 就會自動對應。標註工具可以用 Roboflow、CVAT 或 Label Studio。

### 部署到手機 / 邊緣裝置

```bash
python train.py ... --export tflite   # Android
python train.py ... --export coreml   # iOS
python train.py ... --export ncnn     # 樹莓派
```

## 測試

```bash
pip install pytest
pytest
```

## 專案結構

```
babysleep/
  posture.py   # 關鍵點 → 睡姿規則（純 numpy）
  monitor.py   # 時間平滑、趴睡計時、警報判斷
  detector.py  # YOLO26 推論（pose / detect / classify）
  app.py       # 攝影機擷取 + Flask 網頁監看 + webhook
train.py       # 訓練 / 匯出自訂 YOLO26 模型
data/baby_sleep.yaml
tests/
```
