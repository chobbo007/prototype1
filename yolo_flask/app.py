from flask import Flask, render_template, Response, request
from ultralytics import YOLO
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 로드 (GPU 사용 설정)
model = YOLO("yolov8n.pt").to("cuda")

# 전역 변수
video_path = None
team_model = None
player_colors = []

def get_pure_shirt_color(shirt_area):
    """유니폼 영역에서 잔디색을 제외하고 주요 색상 추출"""
    if shirt_area.size == 0:
        return None
        
    hsv = cv2.cvtColor(shirt_area, cv2.COLOR_BGR2HSV)
    
    # 초록색(잔디) 범위 (H: 35~85 사이)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # 잔디가 아닌 부분만 마스크 생성
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    
    # 색상 추출 (연산 속도를 위해 유효 픽셀만 필터링)
    pure_pixels = hsv[mask_inv > 0]
    
    if len(pure_pixels) > 10:
        # H(색상)와 S(채도)의 평균값 반환
        return np.mean(pure_pixels[:, :2], axis=0)
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    global video_path, team_model, player_colors
    if request.method == "POST":
        # 초기화
        team_model = None
        player_colors = []
        
        file = request.files["video"]
        if file:
            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(video_path)
            return render_template("stream.html")
    return render_template("index.html")

def generate_frames():
    global team_model, player_colors
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # [속도 최적화 1] 2프레임마다 1번만 분석 (실시간성 확보)
        if frame_count % 2 != 0:
            continue

        # [속도 최적화 2] 분석 해상도 최적화 (640이 정확도와 속도의 최적 지점)
        process_frame = cv2.resize(frame, (640, 360))
        
        # YOLO 추론 (verbose=False로 로그 출력 제거하여 속도 향상)
        results = model.predict(process_frame, imgsz=640, conf=0.35, device="cuda", verbose=False)
        
        detections = []

        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # 선수(person) 탐지
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 상체(유니폼) 영역 크롭
                h = y2 - y1
                shirt_img = process_frame[y1 + int(h*0.2):y1 + int(h*0.5), x1:x2]
                
                color_feat = get_pure_shirt_color(shirt_img)
                if color_feat is not None:
                    detections.append((x1, y1, x2, y2, color_feat))
                    
                    # [속도 최적화 3] 학습 데이터 수집 (모델 없을 때만 수행)
                    if team_model is None and len(player_colors) < 100:
                        player_colors.append(color_feat)

        # [팀 자동 분류 학습] 데이터가 80개 이상 쌓이면 최초 1회 실행
        if team_model is None and len(player_colors) >= 80:
            team_model = KMeans(n_clusters=2, n_init=10).fit(player_colors)
            print("--- AI가 팀 유니폼 구분을 완료했습니다 ---")

        # 결과 시각화
        for x1, y1, x2, y2, feat in detections:
            if team_model:
                team = team_model.predict([feat])[0]
                # 팀0: 빨간색, 팀1: 파란색
                box_color = (0, 0, 255) if team == 0 else (255, 0, 0)
            else:
                box_color = (0, 255, 0) # 분류 전 기본 초록색
                
            cv2.rectangle(process_frame, (x1, y1), (x2, y2), box_color, 2)

        # 프레임 전송
        _, buffer = cv2.imencode(".jpg", process_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False) # debug=False로 설정 시 속도 향상