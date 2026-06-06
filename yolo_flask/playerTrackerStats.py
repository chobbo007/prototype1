# PlayerTrackerStats.py
import numpy as np

class PlayerTrackerStats:
    def __init__(self, fps=30):
        self.fps = fps if fps > 0 else 30
        # 선수별 피지컬 스탯 상태 저장소
        # 구조: { track_id: { "last_pos": (x, y), "last_time": float, "total_dist": float, "speed_kmh": float } }
        self.player_memory = {}
        
        # 실제 피치 규격 배율 기준 (FIFA 공식: 105m x 68m)
        # 2D 미니맵 픽셀 규격(1050x680) 기준 -> 10픽셀 = 실제 1미터
        self.PIXELS_PER_METER = 10.0

    def update_player(self, track_id, mini_x, mini_y, frame_idx):
        """
        매 프레임 선수의 미니맵 투영 좌표를 받아 실시간 속도(km/h)와 누적 이동 거리(m)를 계산합니다.
        """
        current_time = frame_idx / self.fps
        
        if track_id not in self.player_memory:
            # 처음 탐지된 선수는 초기화
            self.player_memory[track_id] = {
                "last_pos": (mini_x, mini_y),
                "last_time": current_time,
                "total_dist": 0.0,
                "speed_kmh": 0.0
            }
            return 0.0, 0.0

        stats = self.player_memory[track_id]
        last_x, last_y = stats["last_pos"]
        last_time = stats["last_time"]
        
        # 1. 2D 평면상에서의 프레임 간 픽셀 이동 거리 계산 (피타고라스)
        pixel_dist = np.sqrt((mini_x - last_x)**2 + (mini_y - last_y)**2)
        
        # 2. 픽셀 단위를 실제 물리 미터(m) 단위로 정밀 보정
        meter_dist = pixel_dist / self.PIXELS_PER_METER
        
        # 시간 변화량
        dt = current_time - last_time
        
        if dt > 0 and meter_dist < 15.0:  # 순간 워프나 프레임 튐 현상 방지 필터링 (초당 15m 이하만 인정)
            # 3. 총 누적 거리 계산
            stats["total_dist"] += meter_dist
            
            # 4. 실시간 속도 계산 (m/s -> km/h 변환: 3.6 곱하기)
            speed_mps = meter_dist / dt
            stats["speed_kmh"] = speed_mps * 3.6
            
        # 다음 프레임 연산을 위해 상태 갱신
        stats["last_pos"] = (mini_x, mini_y)
        stats["last_time"] = current_time
        
        return stats["speed_kmh"], stats["total_dist"]

    def generate_final_report_snippet(self):
        """ 경기 종료 후 최고 속도자 및 최다 활동량 선수를 추출하는 텍스트 스니펫 생성 """
        if not self.player_memory:
            return "수집된 피지컬 데이터가 없습니다."
            
        report = "\n🏃‍♂️ [피지컬 데이터 분석 분석 리포트]\n"
        max_speed = -1
        max_speed_player = None
        max_dist = -1
        max_dist_player = None
        
        for tid, data in self.player_memory.items():
            if data["speed_kmh"] > max_speed:
                max_speed = data["speed_kmh"]
                max_speed_player = tid
            if data["total_dist"] > max_dist:
                max_dist = data["total_dist"]
                max_dist_player = tid
                
        report += f" - 최고 스프린트 속도 보유자: Player {max_speed_player} ({max_speed:.1f} km/h)\n"
        report += f" - 최다 활동량(누적 거리) 보유자: Player {max_dist_player} ({max_dist:.1f} m)\n"
        return report