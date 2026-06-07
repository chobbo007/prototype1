import cv2
import torch
import numpy as np
import yaml
import os
import sys
import re
import time
#from flask import Flask, Response, request, render_template, send_from_directory, jsonify
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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

#보로노이 다이어그램 관련 임포트
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPolygon
from sklearn.cluster import KMeans

from matplotlib import rc
#Windows의 맑은 고딕 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

# PlayerBallAssigner, tacticalConvexProcess 인스턴스화
from player_ball_assigner import PlayerBallAssigner
from tacticalConvexProcessor import TacticalConvexProcessor
ball_assigner = PlayerBallAssigner()

# 선수 track 스탯
from playerTrackerStats import PlayerTrackerStats

# Windows의 맑은 고딕 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


app = FastAPI(title="FootBall Tactics Analysis Server")
templates = Jinja2Templates(directory="templates")
app.mount("/uploaded_file", StaticFiles(directory="uploads"), name="uploaded_file")

# --- 1. 모델 로드 및 전역 설정 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("weights/best.pt").to(device)
#yolo_model = YOLO("yolov8x.pt").to(device) #테스트용으로 일반 모델로 변경

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

#=============================== 선수 발 위치 게산해 시각화 도구
# --- [추가] BBox 관련 유틸리티 함수 ---
def get_center_of_bbox(bbox): # 객체의 정중앙 위치 파악에 사용
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox): # 객체의 크기에 비레하는 그래픽을 그리기 위한 기준 값
    return bbox[2] - bbox[0]

def get_foot_position(bbox): # 2차원 화면에서 선수가 밟고 있는 발밑위치를 짚어내기 위함
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

# --- [추가] 고급 시각화 함수 (타원 및 삼각형) ---
def draw_ellipse(frame, bbox, color, track_id=None): # 선수의 발밑에 팀 컬러 색의 3d입체 타원 그리고 id박스 표시하는 시각화 함수
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    
    # 발밑 타원 그리기
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )
    
    # ID 박스 표시
    if track_id is not None:
        rectangle_width, rectangle_height = 40, 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15
        
        cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
        cv2.putText(frame, f"{track_id}", (int(x1_rect + 10), int(y1_rect + 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def draw_triangle(frame, bbox, color): # 머리위에 역삼각현 마커, 게임에서 조작하는 선수나 공의 위치 강조할때의 방식
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)
    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
    return frame
# ============================================= 선수 발 위치 계산하고 시각화 도구

#=============================== 공위치 보간
import pandas as pd

def interpolate_ball_positions(ball_tracking_data):
    """
    ball_tracking_data: list of [frame, id, cls, x1, y1, x2, y2, conf]
    """
    if not ball_tracking_data: return []
    
    # 데이터프레임 변환
    df = pd.DataFrame(ball_tracking_data, columns=['frame', 'id', 'cls', 'x1', 'y1', 'x2', 'y2', 'conf'])
    
    # 프레임 범위 생성 (누락된 프레임 찾기용)
    all_frames = pd.DataFrame({'frame': range(df['frame'].min(), df['frame'].max() + 1)})
    df = pd.merge(all_frames, df, on='frame', how='left')
    
    # 선형 보간 수행 (NaN 값 채우기)
    df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].interpolate(method='linear')
    df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].bfill() # 앞부분 채우기
    
    # 고정값 다시 채우기
    df['id'] = 1
    df['cls'] = 32
    df['conf'] = df['conf'].fillna(0.5)
    
    return df.values.tolist()
# ================================== 공위치 보간
    
def is_valid_player(shirt_area, x_real, y_real, team_model, color_feat):
    """
    [2차 개선: 상대적 판별 로직]
    팀 모델(K-Means)의 결과와 비교하여 심판/골키퍼를 동적으로 필터링
    """
    if shirt_area.size < 10 or color_feat is None: return False
    
    # 1. 무채색 팀(레알 마드리드 등) 대응 로직
    # 팀 모델이 학습된 상태라면, 현재 선수의 색상이 팀 0이나 팀 1의 중심점과 가까운지 확인
    if team_model is not None:
        distances = np.linalg.norm(team_model.cluster_centers_ - color_feat, axis=1)
        min_dist = np.min(distances)
        
        # 만약 두 팀의 대표 색상 모두와 거리가 너무 멀다면 '제3의 인물(심판)'일 확률이 높음
        if min_dist > 50: # 임계값은 히스토그램 스케일에 따라 조정
            return False

    # 2. 코너킥/골문 앞 혼전 상황 대응
    # 골대 근처(abs(x)>48)이더라도, 팀 모델에 의해 팀 컬러로 확정된 상태라면 분석에 포함
    # 즉, 위치만으로 지우는 것이 아니라 '위치 + 팀 컬러 불일치'일 때만 GK로 간주
    if abs(x_real) > 48:
        if team_model is not None:
            # 팀 컬러와의 거리가 가깝다면 위치에 상관없이 우리 선수임
            if min_dist < 30: 
                return True
            else:
                return False # 위치도 끝단인데 팀 컬러도 아니면 골키퍼임
    
    return True

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

#테스트 코드
import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_pure_shirt_color(shirt_area):
    if shirt_area is None or shirt_area.size == 0:
        return None
    
    try:
        h, w, _ = shirt_area.shape

        # [방어코드] 이미지가 너무 작으면 예외 처리
        if h < 5 or w < 5:
            return np.array([shirt_area[:,:,2].mean(), shirt_area[:,:,1].mean(), shirt_area[:,:,0].mean()])

        # 1. BGR -> RGB 및 HSV 동시 변환
        img_rgb = cv2.cvtColor(shirt_area, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(shirt_area, cv2.COLOR_BGR2HSV)

        # 2. 확실한 잔디(초록색) 영역 마스킹 제거 (기존 유저분의 좋았던 아이디어 융합)
        # 축구장 잔디의 일반적인 HSV 범위 (초록색)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # 잔디가 아닌(유니폼 및 피부 등) 픽셀만 추출
        pure_pixels = img_rgb[green_mask == 0]

        # 만약 잔디를 빼고 났더니 남은 픽셀이 너무 없다면 이미지 전체 평균으로 우회
        if len(pure_pixels) < 10:
            return np.array([img_rgb[:,:,0].mean(), img_rgb[:,:,1].mean(), img_rgb[:,:,2].mean()])

        # 3. 남은 픽셀들을 대상으로 KMeans 실행 (유니폼 색상 vs 살색/기타 로고 등 분리 위해 n_clusters=2~3 추천)
        pixel_values = np.float32(pure_pixels)
        
        # 선수의 주된 유니폼 색상을 뽑기 위해 클러스터링
        kmeans = KMeans(n_clusters=2, max_iter=15, random_state=42, n_init=1)
        labels = kmeans.fit_predict(pixel_values)
        centers = kmeans.cluster_centers_

        # 4. 빈도수가 가장 높은 클러스터(가장 지분율이 높은 색상)를 유니폼 대표색으로 선정
        # 잔디를 이미 필터링했기 때문에, 남은 것 중 가장 많은 색이 무조건 유니폼입니다.
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique_labels[np.argmax(counts)]
        
        pure_color = centers[dominant_cluster]
        return pure_color # [R, G, B] 반환

    except Exception as e:
        # 에러 발생 시 부드러운 롤백을 위한 안전장치
        try:
            return np.array([shirt_area[:,:,2].mean(), shirt_area[:,:,1].mean(), shirt_area[:,:,0].mean()])
        except:
            return None
'''
# [개선 3: 색상 특징 추출 고도화]
def get_pure_shirt_color(shirt_area):
    if shirt_area.size == 0: return None
    hsv = cv2.cvtColor(shirt_area, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    pure_pixels = hsv[cv2.bitwise_not(mask) > 0]
    return np.mean(pure_pixels[:, :2], axis=0) if len(pure_pixels) > 5 else None
'''

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

#보로노이
#=====================================================
def draw_pitch_sectors(ax, width=105, height=68):
    # 2번째 이미지 기준 섹터 구분 (단위: m)
    # 가로선: Wide(0~11), Half(11~22), Center(22~46), Half(46~57), Wide(57~68)
    h_lines = [11, 22, 46, 57]
    for hl in h_lines:
        ax.axhline(hl - 34, color='white', linestyle='--', alpha=0.3)
    
    # 세로선: 페널티 박스 기준 등 (필요 시 추가)
    v_lines = [-36, 0, 36] # 예시
    for vl in v_lines:
        ax.axvline(vl, color='white', linestyle='--', alpha=0.3)

def generate_final_report(team_history, filename, ai_text="분석 데이터를 생성중입니다...", create_fifa_pitch=None):
    """
    team_history: {0: [(x,y,frame), ...], 1: [(x,y,frame), ...]} 
    *주의: 프레임별로 팀원들의 위치가 저장되어 있어야 정확한 격자 분석이 가능합니다.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    #폰트 깨짐 방지를 위해 리눅스 공용 기본 폰트를 쓰거나 폰트 설정을 주석처리함
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 격자 데이터 설정 (105m x 68m 경기장을 1m 단위로 분할)
    grid_x, grid_y = np.mgrid[-52.5:52.5:105j, -34:34:68j]

    # 각 격자별로 팀 0이 이긴 횟수와 팀 1이 이긴 횟수를 저장할 공간
    occupancy_map = np.zeros(grid_x.shape) 

    # 프레임별로 그룹화하여 처리 (메모리 효율을 위해)
    # 여기서는 team_history가 전체 좌표 리스트라고 가정하고 평균적인 분포로 근사 계산
    # 실제 프레임별 전수 조사는 연산량이 많으므로 '지배력 알고리즘' 적용
    for t_id in [0, 1]:
        pts = np.array(team_history[t_id])
        if len(pts) == 0: continue

    # 2. 리포트 레이아웃 설정 (왼쪽: 팀0 히트맵 / 가운데: 점유 통계 / 오른쪽: 팀1 히트맵)
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(1, 3, width_ratios=[1, 0.6, 1], wspace=0.25)
    
    pitch_img = create_fifa_pitch()
    pitch_img = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB)

    # 경기장 규격 (FIFA 표준: -52.5 ~ 52.5m, -34 ~ 34m)
    pitch_extent = [-52.5, 52.5, -34, 34]

    # --- [왼쪽: Team 0 누적 히트맵] ---
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(pitch_img, extent=pitch_extent, aspect='auto')

    if len(team_history[0]) > 0:
        pts0 = np.array(team_history[0])
        # 안전장치: 정확히 2열(x, y) 데이터 인지 확인 후에 hexbin 플로팅
        if pts0.ndim == 2 and pts0.shape[1] >= 2:
            ax0.hexbin(pts0[:, 0], pts0[:, 1], gridsize=25, cmap='Reds', alpha=0.6, mincnt=1)

    # 추가 -> 축 범위 고정
    ax0.set_xlim(-52.5, 52.5)
    ax0.set_ylim(-34, 34)
    ax0.set_title("TEAM RED: Activity Heatmap", fontsize=15, pad=10)
    ax0.axis('off')

    # --- [오른쪽: Team 1 누적 히트맵] ---
    ax1 = fig.add_subplot(gs[2])
    ax1.imshow(pitch_img, extent=pitch_extent, aspect='auto')

    if len(team_history[1]) > 0:
        pts1 = np.array(team_history[1])
        if pts1.ndim == 2 and pts1.shape[1] >= 2:
            ax1.hexbin(pts1[:, 0], pts1[:, 1], gridsize=25, cmap='Blues', alpha=0.6, mincnt=1)
    
    # 추가 -> 축 범위 고정
    ax1.set_xlim(-52.5, 52.5)
    ax1.set_ylim(-34, 34)
    ax1.set_title("TEAM BLUE: Activity Heatmap", fontsize=15, pad=10)
    ax1.axis('off')

    # --- [가운데: 평균 점유 영역 비율 (도넛 차트)] ---
    # 실제 보로노이 면적의 누적 평균값을 계산했다고 가정
    area0 = len(team_history[0]) 
    area1 = len(team_history[1])
    total = area0 + area1 if (area0 + area1) > 0 else 1
    
    sizes = [area0/total * 100, area1/total * 100]

    ax_center = fig.add_subplot(gs[1])
    labels = ['Team Red', 'Team Blue']
    colors = ['#ff4d4d','#3399ff']
    
    wedges, texts, autotexts = ax_center.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                            startangle=90, colors=colors, pctdistance=0.85,
                                            textprops={'fontsize': 14, 'weight': 'bold'})
    
    # 도넛 모양 만들기
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax_center.add_artist(centre_circle)
    ax_center.set_title("Average Space Control", fontsize=16, fontweight='bold', pad=20)

    # 하단에 요약 텍스트 추가
    result_text = f"Space Dominance Index\nRED {sizes[0]:.1f}% : BLUE {sizes[1]:.1f}%"
    plt.figtext(0.5, 0.15, result_text, ha='center', fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":10})

    # 파일 저장
    report_path = os.path.join(STATIC_FOLDER, f"report_{filename}.png")
    plt.figtext(0.5, -0.1, "AI Tactical Analysis Summary", ha='center', fontsize=16, fontweight='bold')
    
    import textwrap
    wrapper = textwrap.TextWrapper(width=70)
    wrapped_text = wrapper.fill(text=ai_text)

    plt.figtext(0.5, -0.05, wrapped_text, ha='center', fontsize=12, style='italic', color='#333333')

    plt.subplots_adjust(bottom=0.2)

    plt.savefig(report_path, bbox_inches='tight', dpi=150)
    plt.close()

    return report_path

# 해석을 위한 데이터 추출 예시 (단순 저장용이 아닌 분석용)
def analyze_tactics(full_coords_history):
    report = {}
    for t_id in [0, 1]:
        pts = np.array(full_coords_history[t_id])
        # 1. 중앙 점유율 계산 (x좌표가 -10 ~ 10 사이인 비율)
        mid_control = np.sum((pts[:, 0] > -10) & (pts[:, 0] < 10)) / len(pts)
        report[f"team_{t_id}_mid_control"] = f"{mid_control*100:.1f}%"
        
        # 2. 활동 반경 (표준편차를 이용한 활동 범위)
        activity_radius = np.std(pts, axis=0)
        report[f"team_{t_id}_spread"] = activity_radius.tolist()
    return report


#gemini api호출
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에 등록된 환경 변수들을 로드
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

#gemini api이용
def generate_ai_tactical_report(stats):
    """
    수집된 통계 데이터를 바탕으로 Gemini AI가 전술 분석평을 생성합니다.
    (무료 티어 사용으로 법적/비용 문제 없음)
    """

    try:
        #api 설정
        genai.configure(api_key=gemini_key)
        
        #모델설정
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        당신은 전문 축구 전술 분석가입니다. 제공된 경기 데이터를 분석하여 축구 전문가의 관점에서 종합 분석평을 작성해주세요.
        
        [경기 데이터]
        - 레드팀 점유율: {stats['ratio0']:.1f}%
        - 블루팀 점유율: {stats['ratio1']:.1f}%
        - 레드팀 포메이션: {stats['formation0']}
        - 블루팀 포메이션: {stats['formation1']}
        - 중앙 제어력(레드/블루): {stats['mid_control0']} / {stats['mid_control1']}
        
        [작성 가이드라인]
        1. 데이터 수치에 근거하여 양 팀의 주도권 차이를 설명할 것.
        2. 포메이션에 따른 전술적 특징(예: 공격적 전개, 수비적 역습 등)을 해석할 것.
        3. 일반인이 이해하기 쉽지만 전문적인 용어를 적절히 사용할 것.
        4. 3~4문장 내외의 한국어로 작성할 것.
        """
        
        #콘텐츠 생성
        response = model.generate_content(prompt)

        return response.text
    
    except Exception as e:
        print(f"--- AI 상세 에러로그 시작 ---")
        print(e)
        print(f"--- AI 상세 에러로그 끝 ---")
        return f"AI 분석 중 오류가 발생했습니다: {str(e)}"

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
    
    #[추가] 전술 면적 및 변화율 분석 엔진 활성화
    tactical_visualizer = TacticalConvexProcessor(STATIC_FOLDER, fps, total_to_process)

    #[추가] 선수별 속도 및 이동거리 분석기 인스턴스화
    physical_tracker = PlayerTrackerStats(fps=fps)

    team_model, player_colors, track_history = None, [], {}
    player_ocr_results, latest_H = {}, None
    formation_text = {0: "Waiting...", 1: "Waiting..."}
    
    # 포메이션 라인 유지율 트래커 인스턴스
    area_tracker = TacticalConvexProcessor(STATIC_FOLDER, fps, total_to_process)
    #초당 분석 통꼐 저장을 위한 데이터프레임 구조용 리스트
    per_second_stats = []

    current_f = start_frame
    processed_count = 0

    #보로노이 결과 히트맵용 좌표 저장소
    full_coords_history = {0: [], 1: []}

    confidence_counter = {} # ID별 팀 판정 카운터
    MAX_CONFIRM = 15 #15프레임 이상 일치하면 확정

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

            # === best.pt 적용 시 클래스 매핑 및 안전장치 ===
            class_names = yolo_model.names
            ball_cls = [k for k, v in class_names.items() if 'ball' in v.lower()]
            player_classes = [k for k, v in class_names.items() if 'player' in v.lower() or 'goalkeeper' in v.lower() or 'referee' in v.lower()]

            if not player_classes:
                player_classes = [0]

            # --- [수정 핵심 Step 1] 데이터 분리 수집용 임시 저장소 초기화 ---
            ball_bbox = None
            players_dict = {}       # PlayerBallAssigner에 전달할 딕셔너리
            detected_objects = []   # 이번 프레임의 모든 객체를 모아서 일괄 시각화할 리스트
            team_points = {0: [], 1: []}

            # YOLO 객체 전수조사 및 1차 좌표 분석 순회
            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                rx1, ry1, rx2, ry2 = int(x1 * 1280/w), int(y1 * 720/h), int(x2 * 1280/w), int(y2 * 720/h)
                r_bbox = [rx1, ry1, rx2, ry2]

                # 호모그래피 변환 및 미니맵 좌표 계산
                pt = np.array([(x1 + x2) / 2, y2, 1.0]).reshape(3,1)
                real = np.dot(H_inv, pt); real /= (real[2] + 1e-7)
                px, py = transform_to_minimap(real[0, 0], real[1, 0])
                rx_real, ry_real = real[0, 0], real[1, 0]

                # [A] 공(Ball) 수집 분기
                if cls in ball_cls:
                    ball_bbox = r_bbox  # Assigner용으로 보관
                    detected_objects.append({
                        'type': 'ball', 
                        'bbox': r_bbox, 
                        'px': px, 
                        'py': py
                    })
                    continue

                # [B] 상체 유니폼 추출 및 예외 필터 처리
                shirt_img = frame[max(0, y1+int((y2-y1)*0.2)):min(h, y1+int((y2-y1)*0.5)), max(0,x1):min(w, x2)]
                if shirt_img.size == 0: continue

                color_feat = get_pure_shirt_color(shirt_img)
                if color_feat is None: continue

                is_valid = is_valid_player(shirt_img, rx_real, ry_real, team_model, color_feat)

                if not is_valid:
                    # 심판 또는 골키퍼 정보를 수집 리스트에 추가
                    role_name = 'ref' if 'referee' in class_names[cls].lower() else 'gk'
                    detected_objects.append({
                        'type': 'other', 
                        'role': role_name, 
                        'track_id': track_id, 
                        'bbox': r_bbox, 
                        'px': px, 
                        'py': py
                    })
                    continue

                # [C] 일반 팀 레이블 결정 (시계열 K-Means 알고리즘 적용)
                if track_id not in track_history:
                    if team_model: 
                        predicted_team = team_model.predict([color_feat])[0]
                        if track_id not in confidence_counter:
                            confidence_counter[track_id] = {"team": predicted_team, "count": 1}
                        elif confidence_counter[track_id]["team"] == predicted_team:
                            confidence_counter[track_id]["count"] += 1

                        if confidence_counter[track_id]["count"] >= MAX_CONFIRM:
                            track_history[track_id] = predicted_team
                    else:
                        player_colors.append(color_feat)
                        if len(player_colors) >= 50:
                            team_model = KMeans(n_clusters=2, n_init=20).fit(player_colors)

                label = track_history.get(track_id, -1)
                if label == -1 and track_id in confidence_counter:
                    label = confidence_counter[track_id]["team"]

                # Assigner 전용 데이터셋 구축 (순수 플레이어만 대상)
                players_dict[track_id] = {'bbox': r_bbox}
                
                # 플레이어 수집 리스트에 추가
                detected_objects.append({
                    'type': 'player', 
                    'track_id': track_id, 
                    'bbox': r_bbox, 
                    'label': label,
                    'px': px, 'py': py, 
                    'rx_real': rx_real, 'ry_real': ry_real,
                    'rx1': rx1, 'rx2': rx2, 'ry1': ry1 ,'ry2': ry2
                })

            # --- [수정 핵심 Step 2] 실시간 볼 점유 플레이어 판별 추적 ---
            assigned_player_id = -1
            if ball_bbox is not None and players_dict:
                # 패키지 및 인스턴스 명칭에 맞춰 호출 (글로벌에 ball_assigner 인스턴스가 생성되어 있어야 함)
                assigned_player_id = ball_assigner.assign_ball_to_player(players_dict, ball_bbox)

            # --- [수정 핵심 Step 3] 수집된 데이터를 바탕으로 최종 그래픽 렌더링 ---
            for obj in detected_objects:
                if obj['type'] == 'ball':
                    # 메인 비디오 주황색 삼각형 마커 그리기
                    ball_color = (0, 165, 255)
                    combined_view = draw_triangle(combined_view, obj['bbox'], ball_color)
                    # 미니맵 공 매핑 누락 해결
                    if 0 <= obj['px'] < 1050 and 0 <= obj['py'] < 680:
                        cv2.circle(minimap, (obj['px'], obj['py']), 6, ball_color, -1)

                elif obj['type'] == 'other':
                    # 심판 및 골키퍼 렌더링
                    if obj['role'] == 'ref':
                        combined_view = draw_ellipse(combined_view, obj['bbox'], (0, 255, 255), track_id=f"REF_{obj['track_id']}")
                        if 0 <= obj['px'] < 1050 and 0 <= obj['py'] < 680: cv2.circle(minimap, (obj['px'], obj['py']), 8, (0, 255, 255), -1)
                    else:
                        combined_view = draw_ellipse(combined_view, obj['bbox'], (0, 255, 0), track_id=f"GK_{obj['track_id']}")
                        if 0 <= obj['px'] < 1050 and 0 <= obj['py'] < 680: cv2.circle(minimap, (obj['px'], obj['py']), 8, (0, 255, 0), -1)

                elif obj['type'] == 'player':
                    label = obj['label']
                    p_color = (0,0,255) if label==0 else (255,0,0) if label==1 else (200,200,200)
                    
                    # 미니맵 및 전술 포메이션 연결용 포인트 기록 / px,py는 pnlcalib 호모그래피 통과한 좌표 값
                    px, py = obj['px'], obj['py']

                    #[추가] 호출해서 실시간 속도와 누적거리 획득
                    #  변경할 메인 코드 (실제 미터 좌표 rx_real, ry_real을 넘겨줌)
                    speed_kmh, total_dist = physical_tracker.update_player(
                        obj['track_id'], 
                        obj['rx_real'], 
                        obj['ry_real'], 
                        current_f
                    )

                    #[추가] 메인 영상 선수 머리위에 스텟 바인딩
                    cv2.putText(combined_view, f"{speed_kmh:.1f} km/h", (int(obj['rx1']), int(obj['ry1']) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(combined_view, f"{total_dist:.1f} m", (int(obj['rx1']), int(obj['ry1']) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    # 🌟 공 소유자 전용 시각화 로직 연결!
                    if obj['track_id'] == assigned_player_id:
                        # 볼 소유 플레이어는 발밑 링을 황금색(민트색 등)으로 강조 표시하고 머리 위 역삼각형 마커 추가
                        combined_view = draw_ellipse(combined_view, obj['bbox'], (0, 255, 255), track_id=f"OWNER_{obj['track_id']}")
                        combined_view = draw_triangle(combined_view, obj['bbox'], (0, 255, 255))
                    else:
                        combined_view = draw_ellipse(combined_view, obj['bbox'], p_color, track_id=obj['track_id'])

                    # 미니맵 및 전술 포메이션 연결용 포인트 기록
                    px, py = obj['px'], obj['py']
                    if 0 <= px < 1050 and 0 <= py < 680:
                        cv2.circle(minimap, (px, py), 8, p_color, -1)
                        if label in [0, 1]:
                            team_points[label].append((px, py, obj['rx1'] + (obj['rx2']-obj['rx1'])//2, obj['ry2']))
                    
                    # 보로노이 데이터 수집용 실제 경기장 평면 좌표 적재
                    if label in [0, 1]:
                        if -60 < obj['rx_real'] < 60 and -40 < obj['ry_real'] < 40:
                            full_coords_history[label].append((obj['rx_real'], obj['ry_real']))
            
            # --- 팀별 포메이션(가로/x축) 연결 로직 ---
            for t_id in [0, 1]:
                # ===== 포메이션 라인 유지율 위한 거 =========
                m_pts = [(p[0], p[1]) for p in team_points[t_id]] # team_points에서 미니맵 x, y 픽셀만 분리 추출

                hull, current_area = tactical_visualizer.process_team_hull(m_pts, t_id)
                stability_score = tactical_visualizer.calculate_stability(t_id)

                if hull is not None:
                    poly_color = (0, 0, 255) if t_id == 0 else (255, 0, 0)

                    # 1. 미니맵 위에 팀별 알파 블렌딩 전술 영역 투명 도포 (Convexs Hull 평면화) 
                    overlay = minimap.copy()
                    cv2.drawContours(overlay, [hull], -1, poly_color, -1)
                    # 외각선은 진하게 둘러
                    cv2.drawContours(minimap, [hull], -1, poly_color, 2)
                    alpha = 0.35 #투명도
                    cv2.addWeighted(overlay, alpha, minimap, 1 - alpha, 0, minimap)

                    # 2. 메인 비디오(combined_view) 좌우 패널에 실시간 변화율 데이터 매핑
                    hud_x_pos = 40 if t_id == 0 else 950
                    hud_text_color = (100, 100, 255) if t_id == 0 else (255, 150, 100)

                    #전술 면적 및 대형 유지율 스코어 실시간 출력
                    cv2.putText(combined_view, f"Area: {int(current_area)}px", (hud_x_pos, 660),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_text_color, 2, cv2.LINE_AA)
                    cv2.putText(combined_view, f"Line Hold: {stability_score}%", (hud_x_pos, 660),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_text_color, 2, cv2.LINE_AA)
                #===========================================

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

        #[추가] 매 1초마다 두 팀의 포메이션 데이터 수집하는 if문
        if current_f % fps == 0:
            current_second = (current_f - start_frame) // fps
            a_area = int(area_tracker.area_history[0][-1]) if area_tracker.area_history[0] else 0
            b_area = int(area_tracker.area_history[1][-1]) if area_tracker.area_history[1] else 0
            per_second_stats.append([current_second, a_area, b_area])

        current_f += 1
        processed_count += 1
        analysis_status["progress"] = int((processed_count / total_to_process) * 100)

    # 1. ai에게 전달할 데이터 요약
    tactical_data = analyze_tactics(full_coords_history)

    area0 = len(full_coords_history[0])
    area1 = len(full_coords_history[1])
    total = area0 + area1 if (area0 + area1) > 0 else 1

    stats_for_ai = {
        "ratio0": (area0 / total) * 100,
        "ratio1": (area1 / total) * 100,
        "formation0": formation_text[0],
        "formation1": formation_text[1],
        "mid_control0": tactical_data.get('team_0_mid_control', 'N/A'),
        "mid_control1": tactical_data.get('team_1_mid_control', 'N/A')
    }

    # 2. ai 분석 결과 생성
    ai_commentary = generate_ai_tactical_report(stats_for_ai)

    #리포트 출력 확인용 테스트 ====================
    txt_report_filename = f"report_{filename}.txt"
    txt_report_path = os.path.join(STATIC_FOLDER, txt_report_filename)

    try:
        with open(txt_report_path, "w", encoding="utf-8") as f:
            f.write(ai_commentary)
        print(f"✅ [성공] 전술 분석 텍스트 리포트 저장 완료: {txt_report_path}")
    except Exception as e:
        print(f"❌ [에러] 텍스트 리포트 파일 쓰기 실패: {e}")
    #===========================================

    # [추가] 저장 완료된 텍스트 보고서 파일의 하단에 피지컬 분석 스니펫을 추가
    try:
        with open(txt_report_path, "a", encoding="utf-8") as f:
            f.write(physical_tracker.generate_final_report_snippet())
        print(f"✅ [성공] 피지컬 데이터 요약본 전술 리포트 통합 완료")
    except Exception as e:
        print(f"❌ [에러] 피지컬 리포트 추가 실패: {e}")


    # 3. 최종 상태 업데이트
    #보로노이 최종 리포트 생성
    report_img_path = generate_final_report(full_coords_history, filename, ai_text=ai_commentary, create_fifa_pitch=create_fifa_pitch)

    out.release()
    cap.release()

    # === [추가] 시계열 그래프 및 요약 인사이트 추출 ===
    csv_report_path = os.path.join(STATIC_FOLDER, f"formation_stats_{filename}.csv")
    try:
        import csv
        with open(csv_report_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Second", "Team_Red_Area", "Team_Blue_Area"])
            writer.writerows(per_second_stats)
        print(f"✅ [통계완료] 초당 전술 점유면적 데이터 보존 성공: {csv_report_path}")
    except Exception as e:
        print(f"❌ 통계 데이터 CSV 저장 실패: {e}")
    #===========================================

    #파일명 자르기
    filename = os.path.splitext(filename)[0]
    
    try:
        report_img_path
        print(f"리포트 생성 완료: report_{filename}.png")
    except Exception as e:
        print(f"리포트 생성 에러: {e}")

    analysis_status["progress"] = 100
    analysis_status["complete"], analysis_status["filename"] = True, out_filename

    #보로노이 상태 업데이트
    analysis_status["filename"] = out_filename
    analysis_status["report_img"] = f"report_{filename}.png"
    analysis_status["insight_text"] = ai_commentary
    
    #웹 UI 프론트엔드 및 Flask API에서 자유롭게 꺼내 쓸수 있도록 가공된 원본 메모리 바인딩
    analysis_status["player_physical_records"] = {
        int(k): v for k, v in physical_tracker.player_memory.items()
    }

    #=== [추가] 모듈화된것 포메이션 유지율 한번만 호출 ====
    convex_processor = TacticalConvexProcessor(STATIC_FOLDER, fps, total_to_process)
    g_file, v_file = convex_processor.process(full_coords_history, transform_to_minimap, create_fifa_pitch)

    # 웹 앱 상태 관리 객체 안전 바인딩
    analysis_status["analytics_graph"] = g_file
    analysis_status["convex_video"] = v_file
    #===========================================

# --- 4. FastAPI Routes ---

# [1] GET 요청 처리: 주소창에 그냥 접속했을 때 (index.html 화면 띄우기)
@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    """
    기존 return render_template("index.html")을 대체합니다.
    FastAPI는 인자로 반드시 'request'를 함께 HTML에 넘겨줘야 합니다.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# [2] POST 요청 처리: index.html에서 '비디오 업로드' 버튼을 눌렀을 때
@app.post("/", response_class=HTMLResponse)
async def upload_video_via_form(request: Request, video: UploadFile = File(...)):
    """
    기존 request.files.get('video') 및 file.save(path)를 대체합니다.
    """
    if video and video.filename:
        # 업로드 폴더에 영상 저장 경로 설정
        path = os.path.join(UPLOAD_FOLDER, video.filename)
        
        # FastAPI에서 대용량 파일을 비동기로 안전하게 쓰는 방식
        with open(path, "wb") as buffer:
            buffer.write(await video.read())
            
        # 🌟 중요: stream.html 화면으로 넘어가면서 비디오 파일 이름을 전달
        return templates.TemplateResponse(
            "stream.html", 
            {"request": request, "filename": video.filename}
        )
        
    # 만약 파일이 비어있다면 그냥 다시 메인화면으로 보냄
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/check_status")
async def check_status():
    # 실시간 progress 점수(0~100)와 완성 여부를 JSON 형태로 즉시 반환
    return analysis_status

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)

@app.get("/start_analysis")
async def start_analysis(filename: str, start: float = 0.0, end: float = 0.0, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    백엔드 Spring Boot가 파라미터를 담아 요청하면, 
    FastAPI가 인자를 자동으로 파싱 검증하고 백그라운드 일꾼에게 무거운 YOLO 연산을 안전하게 넘깁니다.
    """
    background_tasks.add_task(run_analysis_batch, filename, start, end)
    return {"status": "started", "message": f"{filename} 분석 전술 대기열 등록 완료."}

if __name__ == "__main__":
    import uvicorn
    # app.py가 파일명이니 "app:app"으로 명시
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
