import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from collections import deque

st.set_page_config(page_title="Crowd Density Detection", layout="centered")
st.title("行人擁擠度偵測 Crowd Density Detection (人/㎡ + ROI + 預警)")

# -------------- Sidebar：安全工程相關參數 --------------
st.sidebar.header("🧭 區域與門檻設定")
use_roi = st.sidebar.checkbox("啟用 ROI（百分比裁切）", value=False,
                              help="若不勾選，使用整個畫面當作監控區域。")

if use_roi:
    col1, col2 = st.sidebar.columns(2)
    x0p = col1.slider("ROI 左(%)", 0, 90, 0, 1)
    y0p = col2.slider("ROI 上(%)", 0, 90, 0, 1)
    x1p = col1.slider("ROI 右(%)", 10, 100, 100, 1)
    y1p = col2.slider("ROI 下(%)", 10, 100, 100, 1)
else:
    x0p, y0p, x1p, y1p = 0, 0, 100, 100

AREA_M2 = st.sidebar.number_input("監控區域實際面積 (㎡)", min_value=1.0, value=20.0, step=1.0,
                                  help="請填實測或圖面估算的區域面積；密度=人數/㎡")

DENSITY_WARN = st.sidebar.number_input("警告門檻 (人/㎡)", min_value=0.5, value=5.0, step=0.5)
DENSITY_DANGER = st.sidebar.number_input("危險門檻 (人/㎡)", min_value=1.0, value=6.5, step=0.5)

HOLD_SECONDS = st.sidebar.number_input("連續超標秒數（觸發預警）", min_value=1, value=5, step=1)
st.sidebar.caption("依 crowd safety 文獻：>5 人/㎡ 高風險、6–7 人/㎡ 極危險；10–30 秒是關鍵反應窗。")

# -------------- 模型 --------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------- 功能函式 --------------
def apply_roi(img, x0p, y0p, x1p, y1p):
    """以百分比裁切 ROI；回傳 ROI 影像與在原圖上的位置"""
    H, W = img.shape[:2]
    x0 = int(W * x0p / 100.0)
    y0 = int(H * y0p / 100.0)
    x1 = int(W * x1p / 100.0)
    y1 = int(H * y1p / 100.0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    roi = img[y0:y1, x0:x1].copy()
    return roi, (x0, y0, x1, y1)

def detect_count_people(img_bgr):
    """回傳 (person_count, boxes)；boxes為(左上右下)座標列表"""
    results = model(img_bgr)
    boxes = results[0].boxes
    person_count = 0
    out_boxes = []
    for box in boxes:
        if int(box.cls[0]) == 0:  # class 0 = person
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out_boxes.append((x1, y1, x2, y2))
    return person_count, out_boxes

def draw_result(base_img, global_boxes, density, level_text, color, roi_rect=None, seconds_over=None):
    img = base_img.copy()
    # ROI 外框
    if roi_rect is not None:
        x0, y0, x1, y1 = roi_rect
        cv2.rectangle(img, (x0, y0), (x1, y1), (180, 180, 180), 2)

    # 人框
    for (x1, y1, x2, y2) in global_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60), 2)

    # 文字
    ybase = 40
    cv2.putText(img, f"Density: {density:.2f} /m^2  [{level_text}]",
                (20, ybase), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    ybase += 35
    if seconds_over is not None:
        cv2.putText(img, f"Over-threshold for: {seconds_over:.1f}s",
                    (20, ybase), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        ybase += 35

    # 密度條
    bar_x, bar_y, bar_w, bar_h = 20, ybase, 320, 26
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
    # Normalize到危險門檻
    ratio = min(density / max(DENSITY_DANGER, 1e-6), 1.0)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + bar_h), color, -1)
    return img

def level_from_density(d):
    if d >= DENSITY_DANGER:
        return "Danger", (0, 0, 255)
    elif d >= DENSITY_WARN:
        return "Warning", (0, 165, 255)
    else:
        return "Normal", (0, 200, 0)

# -------------- 輸入來源 --------------
option = st.radio('選擇輸入來源', ['上傳圖片', '上傳影片', '影片連結', '拍照'])

# -------------- 圖片 --------------
if option == '上傳圖片':
    uploaded_file = st.file_uploader('請上傳圖片', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ROI
        roi_img, (x0, y0, x1, y1) = apply_roi(img_bgr, x0p, y0p, x1p, y1p)
        count, boxes = detect_count_people(roi_img)
        density = count / max(AREA_M2, 1e-6)

        # 將 ROI 內的 boxes 轉回全圖座標
        global_boxes = [(x1_ + x0, y1_ + y0, x2_ + x0, y2_ + y0) for (x1_, y1_, x2_, y2_) in boxes]
        level_text, color = level_from_density(density)
        out = draw_result(img_bgr, global_boxes, density, level_text, color, (x0, y0, x1, y1))

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption='偵測結果（人/㎡）', use_column_width=True)
        st.success(f"ROI 面積 = {AREA_M2:.2f} ㎡,  計數 = {count} 人,  密度 = {density:.2f} 人/㎡,  等級 = {level_text}")

# -------------- 影片（本地上傳/網址）共用迴圈 --------------
def run_video(cap):
    # 估 FPS；若讀不到就假設 30
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_read if fps_read and fps_read > 0 else 30.0

    stframe = st.empty()
    over_hist = deque(maxlen=int(HOLD_SECONDS * fps))
    seconds_over = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi_img, (x0, y0, x1, y1) = apply_roi(frame, x0p, y0p, x1p, y1p)
        count, boxes = detect_count_people(roi_img)
        density = count / max(AREA_M2, 1e-6)
        level_text, color = level_from_density(density)

        # 超標紀錄
        over_hist.append(1 if level_text != "Normal" else 0)
        seconds_over = sum(over_hist) / fps

        global_boxes = [(bx1 + x0, by1 + y0, bx2 + x0, by2 + y0) for (bx1, by1, bx2, by2) in boxes]
        vis = draw_result(frame, global_boxes, density, level_text, color, (x0, y0, x1, y1), seconds_over)

        # 觸發預警（你可在這裡發 API、寫 log、播放聲音等）
        if seconds_over >= HOLD_SECONDS and level_text != "Normal":
            cv2.putText(vis, "ALERT: CROWD RISK!", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        stframe.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels='RGB')

# 影片：本地上傳
elif option == '上傳影片':
    uploaded_video = st.file_uploader('請上傳影片', type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        run_video(cap)
        cap.release()

# 影片：連結
elif option == '影片連結':
    video_url = st.text_input('請貼上影片連結（mp4/avi/mov 直連網址）')
    if video_url:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            st.error('無法開啟影片連結，請確認網址正確且為公開影片。')
        else:
            run_video(cap)
            cap.release()

# 拍照
elif option == '拍照':
    camera_img = st.camera_input('請拍照上傳')
    if camera_img is not None:
        image = Image.open(camera_img).convert('RGB')
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        roi_img, (x0, y0, x1, y1) = apply_roi(img_bgr, x0p, y0p, x1p, y1p)
        count, boxes = detect_count_people(roi_img)
        density = count / max(AREA_M2, 1e-6)

        global_boxes = [(x1_ + x0, y1_ + y0, x2_ + x0, y2_ + y0) for (x1_, y1_, x2_, y2_) in boxes]
        level_text, color = level_from_density(density)
        out = draw_result(img_bgr, global_boxes, density, level_text, color, (x0, y0, x1, y1))

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption='偵測結果（人/㎡）', use_column_width=True)
        st.success(f"ROI 面積 = {AREA_M2:.2f} ㎡,  計數 = {count} 人,  密度 = {density:.2f} 人/㎡,  等級 = {level_text}")
