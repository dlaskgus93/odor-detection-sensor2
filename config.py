"""센서 시스템 설정 파일"""

import os
from pathlib import Path

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models' / 'trained_models'

# 데이터 경로
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
TRAINING_DATA_PATH = DATA_DIR / 'training_data.csv'

# 모델 설정
MODEL_CONFIG = {
    'classifier': {
        'model_type': 'random_forest',  # 'random_forest', 'gradient_boosting', 'cnn'
        'model_path': MODEL_DIR / 'odor_classifier.pkl',
        'n_estimators': 100,
        'max_depth': 15,
        'random_state': 42,
    },
    'regressor': {
        'model_type': 'svr',  # 'linear', 'svr', 'lstm'
        'model_path': MODEL_DIR / 'concentration_predictor.pkl',
        'kernel': 'rbf',
        'C': 100,
        'epsilon': 0.1,
    }
}

# 센서 설정
SENSOR_CONFIG = {
    'gas_sensor': {
        'port': '/dev/ttyUSB0',  # Linux/Mac: '/dev/ttyUSB0', Windows: 'COM3'
        'baudrate': 9600,
        'timeout': 1.0,
        'sampling_rate': 10,  # Hz (샘플링 빈도)
        'buffer_size': 100,   # 데이터 버퍼 크기
    },
    'temperature_sensor': {
        'type': 'DHT22',  # 'DHT22', 'LM35'
        'pin': 4,  # GPIO pin (Raspberry Pi)
        'sampling_rate': 1,  # Hz
    },
    'humidity_sensor': {
        'type': 'DHT22',
        'pin': 4,  # GPIO pin (Raspberry Pi)
        'sampling_rate': 1,  # Hz
    },
}

# 데이터 전처리 설정
PREPROCESSING_CONFIG = {
    'normalize': True,
    'normalization_method': 'minmax',  # 'minmax', 'zscore'
    'remove_outliers': True,
    'outlier_threshold': 3,  # standard deviations
    'smooth_data': True,
    'smoothing_window': 5,
    'fill_missing': True,
    'missing_method': 'interpolate',  # 'interpolate', 'forward_fill'
}

# 특징 추출 설정
FEATURE_CONFIG = {
    'time_domain': [
        'mean', 'std', 'min', 'max', 'median',
        'skewness', 'kurtosis', 'rms', 'peak_to_peak'
    ],
    'frequency_domain': [
        'fft_mean', 'fft_std', 'fft_peaks',
        'spectral_centroid', 'spectral_rolloff'
    ],
    'time_series': [
        'autocorr_lag1', 'autocorr_lag2',
        'trend', 'seasonal_strength'
    ],
    'cross_sensor': [
        'gas_temp_ratio', 'gas_humidity_ratio',
        'temp_humidity_interaction'
    ]
}

# 냄새 라벨
ODOR_LABELS = {
    0: 'Normal',           # 정상
    1: 'Refrigerator',     # 냉장고
    2: 'Industrial',       # 산업현장
    3: 'Decay',            # 부패
    4: 'Chemical',         # 화학물질
    5: 'Smoke',            # 연기
}

# 학습 설정
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'early_stopping': True,
    'early_stopping_patience': 10,
}

# 실시간 감지 설정
REAL_TIME_CONFIG = {
    'window_size': 100,  # 윈도우 크기 (샘플)
    'step_size': 10,     # 슬라이딩 윈도우 스텝
    'update_interval': 1.0,  # 초 단위 업데이트 주기
    'confidence_threshold': 0.7,  # 신뢰도 임계값
    'alert_threshold': 80,  # 알람 임계값 (농도 ppm)
}

# 로깅 설정
LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '{time} | {level} | {name} | {message}',
    'log_file': PROJECT_ROOT / 'logs' / 'odor_detection.log',
}

# 환경 변수 (개발/실제 환경)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')  # 'development', 'production'
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
