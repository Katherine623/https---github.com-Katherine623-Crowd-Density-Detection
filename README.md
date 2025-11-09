# 行人擁擠度偵測系統 Crowd Density Detection

基於 YOLOv8 和 Streamlit 的即時行人密度監控系統，提供人流密度計算（人/㎡）、ROI 區域設定與多級預警功能。

## 📋 功能特色

- **多來源輸入支援**
  - 上傳圖片（JPG/PNG）
  - 上傳影片（MP4/AVI/MOV）
  - 影片連結（串流/直連網址）
  - 即時拍照

- **智慧分析功能**
  - YOLOv8 行人偵測
  - 即時密度計算（人數/㎡）
  - ROI（Region of Interest）區域自訂
  - 三級預警系統（正常/警告/危險）
  - 連續超標時間追蹤

- **視覺化介面**
  - 直覺的 Streamlit Web UI
  - 即時密度顯示條
  - 偵測框標記
  - ROI 區域框線

## 🔧 系統需求

- Python 3.8+
- Windows/Linux/macOS

## 📦 安裝步驟

1. **克隆專案**
```bash
git clone https://github.com/Katherine623/Crowd-Density-Detection.git
cd Crowd-Density-Detection
```

2. **安裝相依套件**
```bash
pip install -r requirements.txt
```

3. **下載 YOLOv8 模型**（如未包含）
```bash
# yolov8n.pt 應已包含在專案中
# 若需重新下載，執行以下指令會自動下載
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 🚀 使用方法

### 啟動應用程式

```bash
streamlit run app.py
```

執行後會自動在瀏覽器開啟 Web 介面（預設 http://localhost:8501）

### 操作說明

#### 1️⃣ 設定監控參數（側邊欄）

- **ROI 設定**：勾選「啟用 ROI」並調整百分比滑桿定義監控區域
- **區域面積**：輸入實際監控區域的面積（㎡）
- **警告門檻**：設定警告密度（預設 5.0 人/㎡）
- **危險門檻**：設定危險密度（預設 6.5 人/㎡）
- **連續超標秒數**：設定觸發預警的持續時間（預設 5 秒）

#### 2️⃣ 選擇輸入來源

- **上傳圖片**：靜態影像分析
- **上傳影片**：本地影片檔案分析
- **影片連結**：支援網路串流或直連影片網址
- **拍照**：使用裝置相機即時拍攝分析

#### 3️⃣ 查看分析結果

- 偵測框會標示每個行人
- 頂部顯示當前密度與等級
- 密度條以顏色表示風險程度
  - 🟢 綠色：正常
  - 🟠 橘色：警告
  - 🔴 紅色：危險
- 當超標時間超過設定秒數時，會顯示「ALERT: CROWD RISK!」

## 📊 密度分級標準

根據群眾安全文獻建議：

| 密度範圍 | 等級 | 風險程度 |
|---------|------|---------|
| < 5.0 人/㎡ | 正常 (Normal) | 低風險 |
| 5.0 - 6.5 人/㎡ | 警告 (Warning) | 高風險 |
| ≥ 6.5 人/㎡ | 危險 (Danger) | 極危險 |

💡 **參考依據**：文獻指出密度 >5 人/㎡ 為高風險，6–7 人/㎡ 為極危險，10–30 秒是關鍵反應窗口。

## 📁 專案結構

```
Crowd-Density-Detection/
├── app.py              # 主程式（Streamlit 應用）
├── requirements.txt    # Python 相依套件
├── yolov8n.pt         # YOLOv8 Nano 模型檔案
└── README.md          # 專案說明文件
```

## 🛠️ 技術棧

- **深度學習框架**：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Web 框架**：[Streamlit](https://streamlit.io/)
- **影像處理**：OpenCV、Pillow
- **數值運算**：NumPy

## 🎯 應用場景

- 🏟️ 大型活動/演唱會人流監控
- 🚇 地鐵站/火車站擁擠度管理
- 🏢 商場/百貨公司客流分析
- 🎓 校園/展覽館安全監控
- 🚨 緊急疏散路線評估

## ⚙️ 進階設定

### 自訂 ROI 區域

ROI（Region of Interest）允許您專注監控特定區域：

1. 勾選側邊欄的「啟用 ROI」
2. 調整左/右/上/下百分比滑桿
3. 系統會在該範圍內進行偵測與計數

### 調整預警參數

根據實際場景調整：

- **人潮密集區**（如地鐵）：建議門檻 4-5 人/㎡
- **一般活動區**（如廣場）：建議門檻 5-6 人/㎡
- **開放空間**：建議門檻 6-8 人/㎡

### 影片串流支援

支援以下格式：

- HTTP/HTTPS 直連影片檔（.mp4/.avi/.mov）
- RTSP 串流（需確認 OpenCV 編譯版本支援）
- YouTube/其他平台（需先取得直連網址）

## 🐛 常見問題

### Q1: 執行時出現「模型載入失敗」

**A:** 確認 `yolov8n.pt` 檔案存在於專案根目錄。可手動下載：
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Q2: 影片連結無法播放

**A:** 請確認：
- 網址為公開且可直接訪問
- 網址指向實際影片檔案（非網頁）
- 網路連線正常

### Q3: 偵測準確度不佳

**A:** 可嘗試：
- 使用更大的 YOLOv8 模型（如 `yolov8m.pt` 或 `yolov8l.pt`）
- 確保影像光線充足、拍攝角度適當
- 調整 ROI 範圍排除干擾區域

### Q4: 記憶體不足

**A:** YOLOv8n 為輕量模型，若仍有問題：
- 降低影片解析度
- 縮小 ROI 範圍
- 確保關閉不必要的背景程式

## 📈 未來改進方向

- [ ] 支援多攝影機多區域同時監控
- [ ] 整合警報通知系統（Email/Webhook）
- [ ] 歷史資料記錄與分析圖表
- [ ] 熱力圖視覺化
- [ ] 支援人流軌跡追蹤
- [ ] 部署為獨立伺服器模式

## 👥 貢獻

歡迎提交 Issue 或 Pull Request！

## 📄 授權

本專案採用 MIT License 授權。

## 📧 聯絡資訊

- GitHub: [@Katherine623](https://github.com/Katherine623)
- 專案連結: [Crowd-Density-Detection](https://github.com/Katherine623/Crowd-Density-Detection)

---

⭐ 如果這個專案對您有幫助，歡迎給個星星支持！
