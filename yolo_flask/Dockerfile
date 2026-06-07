# 1. 모든 환경(일반 PC, GPU 서버, Jetson)에서 호환되는 공용 NVIDIA CUDA 런타임 이미지
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 2. 시스템 필수 패키지 설치 (기본 python3로 변경하여 호환성 확보)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 기본 명령어 연결 규칙 세팅
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 3. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 4. 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . .

# 6. 포트 개방
EXPOSE 8000

# 7. 실행 명령어
CMD ["python", "app.py"]