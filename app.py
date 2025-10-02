import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="Crowd Density Detection", layout="centered")
st.title("行人擁擠度偵測 Crowd Density Detection")

MAX_PERSON = 20

def get_density_level(count, max_count=MAX_PERSON):
    ratio = min(count / max_count, 1.0)
    if ratio <= 0.3:
        return 'Low', (0, 255, 0), ratio  # 綠色
    elif ratio <= 0.6:
        return 'Medium', (0, 255, 255), ratio  # 黃色
    else:
        return 'High', (0, 0, 255), ratio  # 紅色

def detect_and_draw(img, model):
    results = model(img)
    boxes = results[0].boxes
    person_count = 0
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 0:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    level, color, ratio = get_density_level(person_count)
    cv2.putText(img, f'Person Count: {person_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(img, f'Density: {level}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    bar_x, bar_y, bar_w, bar_h = 20, 100, 300, 30
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + bar_h), color, -1)
    return img

model = YOLO('yolov8n.pt')

option = st.radio('選擇輸入來源', ['上傳圖片', '上傳影片', '拍照'])

if option == '上傳圖片':
    uploaded_file = st.file_uploader('請上傳圖片', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        result_img = detect_and_draw(img_bgr, model)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption='偵測結果', use_column_width=True)

elif option == '上傳影片':
    uploaded_video = st.file_uploader('請上傳影片', type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_img = detect_and_draw(frame, model)
            stframe.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels='RGB')
        cap.release()

elif option == '拍照':
    camera_img = st.camera_input('請拍照上傳')
    if camera_img is not None:
        image = Image.open(camera_img).convert('RGB')
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        result_img = detect_and_draw(img_bgr, model)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption='偵測結果', use_column_width=True)
