# Odor Detection Sensor System

반도체식 가스센서(SnO2 nanorods)와 온습도센서를 이용한 냄새 감지 AI 시스템

## 📋 프로젝트 개요

GALD E-beam으로 제작한 SnO2 나노로드 기반 가스센서 어레이와 온습도센서를 활용하여 다양한 냄새(냉장고 냄새, 산업현장 냄새 등)를 미리 학습한 후 냄새의 **종류와 농도를 자동으로 감지**하는 시스템입니다.

## 🎯 주요 기능

- **센서 데이터 수집**: SnO2 gas sensor, 온도센서, 습도센서의 실시간 데이터 획득
- **데이터 전처리**: 정규화, 노이즈 제거, 특징 추출
- **머신러닝 모델**: 냄새 종류 분류 및 농도 예측
- **실시간 감지**: 학습된 모델을 통한 실시간 냄새 감지
- **시각화**: 센서 데이터 및 예측 결과 실시간 모니터링

## 📁 프로젝트 구조

```
odor-detection-sensor2/
├── README.md
├── requirements.txt
├── config.py                 # 설정 파일
├── data/
│   ├── raw/                  # 원본 센서 데이터
│   ├── processed/            # 전처리된 데이터
│   └── training_data.csv     # 학습용 데이터
├── src/
│   ├── __init__.py
│   ├── sensor_data_collector.py    # 센서 데이터 수집
│   ├── data_processor.py           # 데이터 전처리
│   ├── feature_extractor.py        # 특징 추출
│   └── utils.py                    # 유틸리티 함수
├── models/
│   ├── __init__.py
│   ├── odor_classifier.py          # 냄새 분류 모델
│   ├── concentration_predictor.py   # 농도 예측 모델
│   └── trained_models/             # 저장된 모델
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_real_time_detection.ipynb
├── scripts/
│   ├── collect_training_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── real_time_detection.py
└── tests/
    ├── test_data_processor.py
    └── test_models.py
```

## 🛠️ 설치 및 설정

### 1. 저장소 클론
```bash
git clone https://github.com/turtlemuziq/odor-detection-sensor2.git
cd odor-detection-sensor2
```

### 2. 가상 환경 생성
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 필요 라이브러리 설치
```bash
pip install -r requirements.txt
```

## 📊 데이터 수집 및 학습

### Step 1: 센서 데이터 수집
```bash
python scripts/collect_training_data.py --duration 3600 --odor_type "refrigerator"
```

### Step 2: 모델 훈련
```bash
python scripts/train_models.py --epochs 100 --batch_size 32
```

### Step 3: 모델 평가
```bash
python scripts/evaluate_models.py --test_size 0.2
```

### Step 4: 실시간 냄새 감지
```bash
python scripts/real_time_detection.py --model_path models/trained_models/classifier.pkl
```

## 🔧 센서 설정

### 지원하는 센서
- **가스센서**: SnO2 nanorods (GALD E-beam 제작)
- **온도센서**: DHT22 또는 LM35
- **습도센서**: DHT22
- **인터페이스**: Serial/I2C/SPI

### config.py 설정 예시
```python
SENSOR_CONFIG = {
    'gas_sensor': {
        'port': '/dev/ttyUSB0',  # 또는 'COM3' (Windows)
        'baudrate': 9600,
        'sampling_rate': 10,  # Hz
    },
    'temperature_sensor': {'pin': 4},  # GPIO pin
    'humidity_sensor': {'pin': 4},     # GPIO pin
    'calibration_time': 60,  # seconds
}
```

## 🧠 머신러닝 모델

### 냄새 분류 모델 (Classification)
- **알고리즘**: Random Forest / Gradient Boosting / CNN
- **입력**: 센서 신호 특징 (통계, FFT, 시계열 특징)
- **출력**: 냄새 종류 (냉장고, 산업현장, 정상 등)

### 농도 예측 모델 (Regression)
- **알고리즘**: Linear Regression / SVR / LSTM
- **입력**: 센서 신호
- **출력**: 냄새 농도 (0-100 ppm 또는 상대 농도)

## 📈 특징 추출 (Feature Engineering)

```python
- 시간 영역 특징: 평균, 표준편차, 최댓값, 최솟값, RMS
- 주파수 영역 특징: FFT, 전력스펙트럼
- 시계열 특징: 자기상관, 변화율
- 센서 간 상호작용: 센서 간 비율, 조합 지수
```

## 🔍 실시간 모니터링

### 웹 대시보드 (선택사항)
```bash
python -m streamlit run dashboards/monitoring_dashboard.py
```

## 📚 참고 자료

- SnO2 Gas Sensors: [relevant papers]
- Time Series Feature Engineering: [references]
- Real-time ML Inference: [resources]

## 🤝 기여

Issues와 Pull Requests를 환영합니다!

## 📄 라이선스

MIT License

## 📧 연락처

프로젝트에 대한 질문이나 제안이 있으신가요? Issues를 통해 연락주세요.
