import os
import sys
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TacticalConvexProcessor:
    def __init__(self, static_folder, fps=30, total_to_process=1):
        self.static_folder = static_folder
        self.fps = fps if fps > 0 else 30
        self.total_to_process = total_to_process if total_to_process > 0 else 1
        
        # app.py 실시간 수집부 호환용 변수
        self.area_history = {0: [0.0], 1: [0.0]}
        
    def process_team_hull(self, mini_pts, team_id):
        """ app.py 메인 루프 호환용 실시간 Hull 연산기 """
        if len(mini_pts) >= 3:
            hull = cv2.convexHull(np.array(mini_pts, dtype=np.int32).reshape(-1, 1, 2))
            area = cv2.contourArea(hull)
            self.area_history[team_id].append(area)
            return hull, area
        else:
            self.area_history[team_id].append(0.0)
            return None, 0.0

    def calculate_stability(self, team_id):
        """ app.py 메인 루프 호환용 라인 유지 점수 """
        return 85.0

    def _draw_ui_panel(self, img, team_name, dm_area, ma_area, color, is_left_pos):
        """ 2dxyz4.py의 핵심 기능: 세련된 실시간 HUD 패널 그래픽 렌더링 """
        h, w = img.shape[:2]
        panel_w = 260
        panel_h = 110
        
        # 좌/우 패널 여백 좌표 설정
        x1 = 20 if is_left_pos else w - panel_w - 20
        y1 = h - panel_h - 20
        x2, y2 = x1 + panel_w, y1 + panel_h
        
        # 반투명 배경 박스 (알파 블렌딩)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # 텍스트 HUD 메트릭 디자인
        cv2.putText(img, f"📊 {team_name} HUD", (x1 + 15, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"DEF-MID Space: {int(dm_area)} px", (x1 + 15, y1 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"MID-ATT Space: {int(ma_area)} px", (x1 + 15, y1 + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    def process(self, full_coords_history, transform_to_minimap, create_fifa_pitch):
        """
        [3단 종진 분할 알고리즘 + 개별 면적 측정 + HUD 연동 완공 버전]
        """
        print("🚀 [독립 프로세스] 포메이션 3단 분할 및 실시간 HUD 비디오 생성을 시작합니다.")
        
        convex_video_path = os.path.join(self.static_folder, "tactical_convex_video.mp4")
        convex_out = cv2.VideoWriter(convex_video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1050, 680))
        
        # 시계열 통계 저장용 리스트
        ts_dm_0, ts_ma_0 = [], []
        ts_dm_1, ts_ma_1 = [], []
        
        max_loop = max(len(full_coords_history[0]), len(full_coords_history[1]))
        if max_loop == 0:
            print("⚠️ 수집된 평면 좌표 데이터가 없어 분석을 건너뜁니다.")
            return "analytics_timeline.png", "tactical_convex_video.mp4"
            
        # 속도 부스팅 샘플링 (초당 2프레임 수준 연산)
        sample_step = max(1, self.fps // 2)
        
        for curr_idx in range(0, max_loop, sample_step):
            pure_pitch = create_fifa_pitch() # 1050x680 축구장 템플릿
            
            # --- [팀 0 및 팀 1 공통 처리 루프] ---
            current_spaces = {0: {"dm": 0.0, "ma": 0.0}, 1: {"dm": 0.0, "ma": 0.0}}
            
            for t_id, color in zip([0, 1], [(0, 0, 255), (255, 0, 0)]):
                sub_pts = full_coords_history[t_id][:curr_idx+1][-11:] # 최근 선수 좌표 샘플링
                mini_pts = []
                
                for rx, ry in sub_pts:
                    px, py = transform_to_minimap(rx, ry)
                    if 0 <= px < 1050 and 0 <= py < 680:
                        mini_pts.append([px, py])
                        cv2.circle(pure_pitch, (px, py), 5, color, -1)
                
                # ⭐ 기능 이식 1 & 2: K-Means 기반 3단 포메이션 분할 및 개별 면적 연산
                if len(mini_pts) >= 4:
                    pts_arr = np.array(mini_pts, dtype=np.float32)
                    x_coords = pts_arr[:, 0].reshape(-1, 1)
                    
                    # X축(종진) 기준으로 3개 그룹(수비, 미드, 공격) 군집 분류
                    k = min(3, len(mini_pts))
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(x_coords, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    
                    # 센터 좌표 기준으로 정렬하여 DEF, MID, ATT 그룹 매핑
                    centers_sorted_idx = np.argsort(centers.flatten())
                    
                    def_pts, mid_pts, att_pts = [], [], []
                    for i, label in enumerate(labels.flatten()):
                        p = mini_pts[i]
                        group_rank = np.where(centers_sorted_idx == label)[0][0]
                        if group_rank == 0: def_pts.append(p)
                        elif group_rank == 1: mid_pts.append(p)
                        else: att_pts.append(p)
                    
                    # 1) 수비 - 미드필더 간격 다각형 면적 구하기 (DEF-MID Space)
                    dm_group = def_pts + mid_pts
                    if len(dm_group) >= 3:
                        hull_dm = cv2.convexHull(np.array(dm_group, dtype=np.int32))
                        overlay = pure_pitch.copy()
                        cv2.drawContours(overlay, [hull_dm], -1, color, -1)
                        cv2.addWeighted(overlay, 0.15, pure_pitch, 0.85, 0, pure_pitch) # 연한 채우기
                        current_spaces[t_id]["dm"] = cv2.contourArea(hull_dm)
                    
                    # 2) 미드필더 - 공격수 간격 다각형 면적 구하기 (MID-ATT Space)
                    ma_group = mid_pts + att_pts
                    if len(ma_group) >= 3:
                        hull_ma = cv2.convexHull(np.array(ma_group, dtype=np.int32))
                        overlay = pure_pitch.copy()
                        cv2.drawContours(overlay, [hull_ma], -1, color, -1)
                        cv2.addWeighted(overlay, 0.25, pure_pitch, 0.75, 0, pure_pitch) # 조금 더 진한 채우기
                        current_spaces[t_id]["ma"] = cv2.contourArea(hull_ma)
            
            # 시계열 통계 데이터 적재
            ts_dm_0.append(current_spaces[0]["dm"])
            ts_ma_0.append(current_spaces[0]["ma"])
            ts_dm_1.append(current_spaces[1]["dm"])
            ts_ma_1.append(current_spaces[1]["ma"])
            
            # ⭐ 기능 이식 3: 실시간 HUD 정보 패널 좌우 하단 투포에 도포
            self._draw_ui_panel(pure_pitch, "TEAM RED", current_spaces[0]["dm"], current_spaces[0]["ma"], (0, 0, 255), is_left_pos=True)
            self._draw_ui_panel(pure_pitch, "TEAM BLUE", current_spaces[1]["dm"], current_spaces[1]["ma"], (255, 0, 0), is_left_pos=False)
            
            # 비디오 프레임 속도 맞춤용 중복 쓰기 복사
            for _ in range(sample_step):
                convex_out.write(pure_pitch)
                
        convex_out.release()
        print("✅ [기능추가 완공 1] 3단 분할 전술 비디오 인코딩 완료: tactical_convex_video.mp4")
        
        # 2. 고도화된 세부 시계열 라인별 변동 그래프 작성
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        axes[0].plot(ts_dm_0, color='red', linestyle='-', label='RED DEF-MID Space')
        axes[0].plot(ts_dm_1, color='blue', linestyle='-', label='BLUE DEF-MID Space')
        axes[0].set_title("🕒 Time-Series DEF-MID Line Distance Trend")
        axes[0].set_ylabel("Pixel Area")
        axes[0].legend()
        
        axes[1].plot(ts_ma_0, color='darkred', linestyle='--', label='RED MID-ATT Space')
        axes[1].plot(ts_ma_1, color='darkblue', linestyle='--', label='BLUE MID-ATT Space')
        axes[1].set_title("📊 Time-Series MID-ATT Line Distance Trend")
        axes[1].set_xlabel("Tactical Timeline Flows (Sampled)")
        axes[1].set_ylabel("Pixel Area")
        axes[1].legend()
        
        plt.tight_layout()
        graph_out_path = os.path.join(self.static_folder, "analytics_timeline.png")
        plt.savefig(graph_out_path, dpi=150)
        plt.close()
        print("✅ [기능추가 완공 2] 라인별 세부 시계열 분석 그래프 완료: analytics_timeline.png")
        
        # 3. 디테일 요약 텍스트 리포트 생성
        summary_out_path = os.path.join(self.static_folder, "tactical_summary.txt")
        with open(summary_out_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("    축구 전술 분석 포메이션 라인별 간격 상세 보고서    \n")
            f.write("==================================================\n\n")
            f.write(f"1. TEAM RED (레드팀 간격 분석)\n")
            f.write(f" - 평균 수비-미드필더 간격 격차: {int(np.mean(ts_dm_0))} px\n")
            f.write(f" - 평균 미드필더-공격수 간격 격차: {int(np.mean(ts_ma_0))} px\n\n")
            f.write(f"2. TEAM BLUE (블루팀 간격 분석)\n")
            f.write(f" - 평균 수비-미드필더 간격 격차: {int(np.mean(ts_dm_1))} px\n")
            f.write(f" - 평균 미드필더-공격수 간격 격차: {int(np.mean(ts_ma_1))} px\n\n")
            
        print("✅ [기능추가 완공 3] 세부 텍스트 요약 리포트 완료: tactical_summary.txt")
        return "analytics_timeline.png", "tactical_convex_video.mp4"