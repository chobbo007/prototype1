#import sys 
#sys.path.append('../')
#from utils import get_center_of_bbox, measure_distance

def get_center_of_bbox(bbox):
    """바운딩 박스의 중심점 (X, Y) 좌표를 반환"""
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

def measure_distance(p1, p2):
    """두 점 사이의 유클리드 거리를 계산"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

#현재 축구장위에서 공을 소유하고 있는 선수가 누구인지 판별하는 시스템
'''
[동작 순서]
1. 현재 프레임에서 공의 중심점(X, Y) 좌표를 구한다.
2. 경기장 내에 있는 모든 선수를 한 명씩 조사한다.
3. 선수의 발 위치(바운딩 박스의 하단)와 공 사이의 거리를 계산한다.
4. 설정해둔 커트라인(70픽셀)보다 가까우면서, 그중에서도 "가장 공과 가까운 선수"를 소유자로 최종 낙점한다.
'''
class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            #선수 바운딩 박스에서 왼발위치
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            #선수 바운딩 박스에서 오른발위치
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player