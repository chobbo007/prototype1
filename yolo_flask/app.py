import cv2
import torch
import numpy as np
import yaml
import os
import sys
import re
import time
from flask import Flask, Response, request, render_template, send_from_directory, jsonify
from ultralytics import YOLO
from sklearn.cluster import KMeans
from threading import Thread
from queue import Queue
from collections import deque
import torchvision.transforms as T
from paddleocr import PaddleOCR

# PnLCalib 관련 (기존 임포트 유지)
from inference import get_cls_net, get_cls_net_l, inference, projection_from_cam_params
from utils.utils_calib import FramebyFrameCalib
import inference as inf_module

app = Flask(__name__)

# --- 1. 모델 로드 및 전역 설정 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolov8n.pt").to(device)

try:
    ocr = PaddleOCR(lang='en', enable_mkldnn=False, show_log=False)
except Exception as e:
    print(f"OCR 로드 에러: {e}")
    ocr = None

# PnL 설정
cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))
model_kp = get_cls_net(cfg).to(device).eval()
model_kp.load_state_dict(torch.load("weights/SV_kp.pth", map_location=device))
model_line = get_cls_net_l(cfg_l).to(device).eval()
model_line.load_state_dict(torch.load("weights/SV_lines.pth", map_location=device))
inf_module.device, inf_module.transform2, inf_module.model_l = device, T.Resize((540, 960)), model_line

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# 전역 상태 변수
analysis_status = {"progress": 0, "complete": False, "filename": ""}

# --- 2. 유틸리티 함수 ---
def create_fifa_pitch(width=1050, height=680):
    # FIFA 정규 규격 비율에 맞춰 미니맵 배경 생성하기
    # 1. 잔디색 배경
    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)

    line_color = (255, 255, 255) # 흰색 라인
    thickness = 5

    # 2. 외곽 라인
    cv2.rectangle(pitch, (0, 0), (width, height), line_color, thickness)

    # 3. 하프라인 및 센터서클
    cv2.line(pitch, (width // 2, 0), (width // 2, height), line_color, thickness)
    cv2.circle(pitch, (width // 2, height // 2), 92, line_color, thickness) # 반지름 약 9.15m
    cv2.circle(pitch, (width // 2, height // 2), 3, line_color, -1) # 센터 점

    # 4. 페널티 박스 (16.5m 영역)
    box_w, box_h = 165, 402
    top_y = (height - box_h) // 2
    # 왼쪽 / 오른쪽
    cv2.rectangle(pitch, (0, top_y), (box_w, top_y + box_h), line_color, thickness)
    cv2.rectangle(pitch, (width - box_w, top_y), (width, top_y + box_h), line_color, thickness)

    # 5. 골 에어리어 (5.5m 영역)
    g_box_w, g_box_h = 55, 182
    g_top_y = (height - g_box_h) // 2
    cv2.rectangle(pitch, (0, g_top_y), (g_box_w, g_top_y + g_box_h), line_color, thickness)
    cv2.rectangle(pitch, (width - g_box_w, g_top_y), (width, g_top_y + g_box_h), line_color, thickness)

    # 6. 아크 서클 (페널티 아크)
    cv2.ellipse(pitch, (110, height // 2), (92, 92), 0, -53, 53, line_color, thickness)
    cv2.ellipse(pitch, (width - 110, height // 2), (92, 92), 0, 127, 233, line_color, thickness)

    return pitch

def transform_to_minimap(x_real, y_real):
    return int((x_real + 52.5) * 10), int((y_real + 34.0) * 10)

def get_pure_shirt_color(shirt_area):
    if shirt_area.size == 0: return None
    hsv = cv2.cvtColor(shirt_area, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    pure_pixels = hsv[cv2.bitwise_not(mask) > 0]
    return np.mean(pure_pixels[:, :2], axis=0) if len(pure_pixels) > 5 else None

# [추가/수정] X축 기준 가로 그룹화 알고리즘
def get_formation_groups_horizontal(pts):
    if len(pts) < 3: return [], "Analyzing..."
    
    # 1. X좌표(가로) 기준 정렬
    pts_sorted = sorted(pts, key=lambda p: p[0])
    x_min, x_max = pts_sorted[0][0], pts_sorted[-1][0]
    x_range = max(1, x_max - x_min)
    
    # 2. X축 비율로 구역 나눔 (왼쪽 35%, 오른쪽 35%, 나머지 중앙)
    l_zone = [p for p in pts_sorted if p[0] <= x_min + x_range * 0.35]
    r_zone = [p for p in pts_sorted if p[0] >= x_max - x_range * 0.35]
    c_zone = [p for p in pts_sorted if p not in l_zone and p not in r_zone]
    
    # 3. 각 구역 내에서는 Y좌표(세로) 순서대로 정렬하여 가로 라인이 형성되게 함
    groups = [
        sorted(l_zone, key=lambda p: p[1]),
        sorted(c_zone, key=lambda p: p[1]),
        sorted(r_zone, key=lambda p: p[1])
    ]
    
    formation_name = f"L{len(l_zone)}-C{len(c_zone)}-R{len(r_zone)}"
    return groups, formation_name

# --- 3. 핵심 분석 엔진 ---
def run_analysis_batch(filename, start_t, end_t):
    global analysis_status
    analysis_status = {"progress": 0, "complete": False, "filename": ""}
    
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(float(start_t) * fps)
    end_frame = int(float(end_t) * fps) if float(end_t) > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_to_process = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_filename = f"result_video.mp4"
    output_path = os.path.join(STATIC_FOLDER, out_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    cam_calib = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
    pitch_bg = create_fifa_pitch()
    
    team_model, player_colors, track_history = None, [], {}
    player_ocr_results, latest_H = {}, None
    formation_text = {0: "Waiting...", 1: "Waiting..."}
    
    current_f = start_frame
    processed_count = 0

    while cap.isOpened() and current_f <= end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        if current_f % 30 == 0 or latest_H is None:
            res = inference(cam_calib, cv2.resize(frame, (640, 360)), model_kp, model_line, 0.3434, 0.7867, pnl_refine=False)
            if res:
                P = projection_from_cam_params(res)
                latest_H = P[:, [0, 1, 3]]

        results = yolo_model.track(frame, imgsz=960, persist=True, verbose=False, tracker="bytetrack.yaml")
        combined_view = cv2.resize(frame, (1280, 720))
        minimap = pitch_bg.copy()

        if latest_H is not None and results[0].boxes.id is not None:
            H_inv = np.linalg.inv(latest_H)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            team_points = {0: [], 1: []}

            for box, track_id, cls in zip(boxes, ids, classes):
                if cls != 0: continue
                x1, y1, x2, y2 = map(int, box)
                rx1, ry1, rx2, ry2 = int(x1 * 1280/w), int(y1 * 720/h), int(x2 * 1280/w), int(y2 * 720/h)

                shirt_img = frame[max(0, y1+int((y2-y1)*0.2)):min(h, y1+int((y2-y1)*0.5)), max(0,x1):min(w, x2)]
                
                if track_id not in track_history and shirt_img.size > 0:
                    color_feat = get_pure_shirt_color(shirt_img)
                    if color_feat is not None:
                        if team_model: track_history[track_id] = team_model.predict([color_feat])[0]
                        else:
                            player_colors.append(color_feat)
                            if len(player_colors) >= 30: team_model = KMeans(n_clusters=2, n_init=10).fit(player_colors)

                label = track_history.get(track_id, -1)
                color = (0,0,255) if label==0 else (255,0,0) if label==1 else (200,200,200)
                
                cv2.rectangle(combined_view, (rx1, ry1), (rx2, ry2), color, 2)
                
                pt = np.array([(x1 + x2) / 2, y2, 1.0]).reshape(3, 1)
                real = np.dot(H_inv, pt); real /= (real[2] + 1e-7)
                px, py = transform_to_minimap(real[0, 0], real[1, 0])
                
                if 0 <= px < 1050 and 0 <= py < 680:
                    cv2.circle(minimap, (px, py), 8, color, -1)
                    if label in [0, 1]:
                        team_points[label].append((px, py, rx1 + (rx2-rx1)//2, ry2))

            # --- 팀별 포메이션(가로/x축) 연결 로직 ---
            for t_id in [0, 1]:
                pts = team_points[t_id]
                groups, f_name = get_formation_groups_horizontal(pts)
                formation_text[t_id] = f_name
                
                #라인 색상 설정
                line_col = (0, 255, 255) if t_id == 0 else (255, 255, 0)
                for group in groups:
                    if len(group) >= 2:
                        for i in range(len(group)-1):
                            cv2.line(combined_view, (group[i][2], group[i][3]),
                                     (group[i+1][2], group[i+1][3]), line_col, 2)
                            
                            #미니맵에 포메이션 표시한다맨이야.
                            cv2.line(minimap,
                                     (group[i][0], group[i][1]),
                                     (group[i+1][0], group[i+1][1]),
                                     line_col, 1, cv2.LINE_AA)
                #팀별로 포메이션 텍스트 표시맨
                y_pos = 650 if t_id == 0 else 50
                cv2.putText(minimap, f"Team {t_id}: {formation_text[t_id]}", (50, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        mini_h, mini_w = 200, 300
        mini_resized = cv2.resize(minimap, (mini_w, mini_h))
        combined_view[20:20+mini_h, 1280-mini_w-20:1280-20] = mini_resized

        out.write(combined_view)
        current_f += 1
        processed_count += 1
        analysis_status["progress"] = int((processed_count / total_to_process) * 100)

    out.release()
    cap.release()
    analysis_status["complete"], analysis_status["filename"] = True, out_filename

# --- 4. Flask Routes ---
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            return render_template("stream.html", filename=file.filename)
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/start_analysis")
def start_analysis():
    filename = request.args.get("filename")
    start = request.args.get("start", 0)
    end = request.args.get("end", 0)
    Thread(target=run_analysis_batch, args=(filename, start, end), daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/check_status")
def check_status():
    return jsonify(analysis_status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)