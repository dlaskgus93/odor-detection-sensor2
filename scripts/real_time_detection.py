#!/usr/bin/env python3
"""실시간 냄새 감지 시스템"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import deque
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 프로젝트 루트로 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensor_data_collector import SensorDataCollector
from src.data_processor import DataProcessor
from src.feature_extractor import FeatureExtractor
from models.odor_classifier import OdorClassifier
from models.concentration_predictor import ConcentrationPredictor
from src.utils import ensure_directory_exists
import config


class RealTimeOdorDetector:
    """실시간 냄새 감지 시스템"""
    
    def __init__(self, use_real_sensor: bool = False, window_size: int = 100):
        self.use_real_sensor = use_real_sensor
        self.window_size = window_size
        
        self.sensor_collector = SensorDataCollector(config.SENSOR_CONFIG)
        self.data_processor = DataProcessor(config.PREPROCESSING_CONFIG)
        self.feature_extractor = FeatureExtractor(config.FEATURE_CONFIG)
        
        self.classifier = self._load_classifier()
        self.regressor = self._load_regressor()
        
        self.data_buffer = deque(maxlen=window_size)
        self.detection_history = deque(maxlen=100)
        self.concentration_history = deque(maxlen=100)
        
        self.logs_dir = Path(__file__).parent.parent / 'logs' / 'detections'
        ensure_directory_exists(self.logs_dir)
        print("✅ 실시간 냄새 감지 시스템 초기화 완료")
    
    def _load_classifier(self) -> OdorClassifier:
        path = config.MODEL_DIR / 'odor_classifier.pkl'
        if not path.exists():
            print("⚠️ 분류 모델이 없습니다. 먼저 훈련해주세요.")
            sys.exit(1)
        return OdorClassifier.load(path)
    
    def _load_regressor(self) -> ConcentrationPredictor:
        path = config.MODEL_DIR / 'concentration_predictor.pkl'
        if not path.exists(): return None
        return ConcentrationPredictor.load(path)
    
    def detect_odor(self, verbose: bool = True) -> dict:
        sensor_data = self.sensor_collector.read_all_sensors()
        self.data_buffer.append(sensor_data)
        
        if len(self.data_buffer) < self.window_size:
            return {'status': 'buffering', 'buffered': len(self.data_buffer), 'required': self.window_size}
        
        buffer_df = pd.DataFrame(list(self.data_buffer))
        processed_df = self.data_processor.preprocess_dataframe(buffer_df)
        
        features = self.feature_extractor.extract_all_features(
            gas_data=processed_df['gas'].values,
            temp_data=processed_df['temperature'].values,
            humidity_data=processed_df['humidity'].values,
        )
        
        odor_proba = self.classifier.predict_proba(features.values)
        odor_pred = self.classifier.predict(features.values)[0]
        
        # --- 수정된 부분: AI가 뱉은 숫자를 다시 이름으로 번역 ---
        label_names = ['chemical', 'decay', 'industrial', 'normal', 'refrigerator', 'smoke']
        if isinstance(odor_pred, (int, np.integer)) and odor_pred < len(label_names):
            odor_label = label_names[odor_pred]
        else:
            odor_label = str(odor_pred)
        # -----------------------------------------------------------
        
        confidence = odor_proba[0].max()
        
        concentration = None
        if self.regressor:
            concentration = max(0, min(100, self.regressor.predict(features.values)[0]))
            
        result = {
            'status': 'detected',
            'timestamp': datetime.now().isoformat(),
            'odor_type': odor_label,
            'confidence': float(confidence),
            'concentration': float(concentration) if concentration else None,
            'temperature': float(processed_df['temperature'].mean()),
            'humidity': float(processed_df['humidity'].mean())
        }
        
        self.detection_history.append(result)
        return result
    
def monitor_continuous(self, update_interval: float = 2.0, alert_threshold: float = None, save_log: bool = False):
        print("\n" + "#"*70 + "\n📡 실시간 모니터링 시작 (Ctrl+C로 종료)\n" + "#"*70)
        
        # --- 💡 수정된 부분: 모니터링을 시작할 때 '시분초'가 포함된 고유한 파일명을 1번만 만듭니다. ---
        session_log_file = None
        if save_log:
            session_time = datetime.now().strftime('%Y%m%d_%H%M%S') # 예: 20260531_174530
            session_log_file = self.logs_dir / f"detections_{session_time}.jsonl"
            print(f"📝 이번 모니터링 기록은 다음 파일에 저장됩니다: {session_log_file.name}\n")
        # -------------------------------------------------------------------------
            
        try:
            while True:
                res = self.detect_odor(verbose=False)
                if res.get('status') == 'buffering':
                    print(f"\r[데이터 수집 중] {res['buffered']}/{res['required']}", end='', flush=True)
                    time.sleep(0.5)
                    continue
                
                msg = f"\r[{datetime.now().strftime('%H:%M:%S')}] 냄새: {str(res['odor_type']).upper():12} | 신뢰도: {res['confidence']:.1%}"
                if res['concentration']:
                    msg += f" | 농도: {res['concentration']:5.1f} ppm"
                    if alert_threshold and res['concentration'] > alert_threshold:
                        msg += " ⚠️ 경고! 농도 초과!"
                        
                print(msg + " " * 10) # 덮어쓰기 잔상 제거를 위한 여백
                
                # --- 💡 수정된 부분: 미리 만들어둔 파일명에 데이터를 안전하게 추가합니다. ---
                if save_log and session_log_file:
                    with open(session_log_file, 'a') as f:
                        f.write(json.dumps(res, ensure_ascii=False) + '\n')
                # ----------------------------------------------------------------
                        
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n모니터링을 종료합니다.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor', action='store_true', help='실시간 모니터링 모드')
    parser.add_argument('--use_real_sensor', action='store_true')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--update_interval', type=float, default=2.0)
    parser.add_argument('--alert_threshold', type=float)
    parser.add_argument('--save_log', action='store_true', help='결과를 파일로 저장합니다')
    args = parser.parse_args()
    
    try:
        detector = RealTimeOdorDetector(use_real_sensor=args.use_real_sensor, window_size=args.window_size)
        if args.monitor:
            detector.monitor_continuous(
                update_interval=args.update_interval, 
                alert_threshold=args.alert_threshold,
                save_log=args.save_log
            )
        else:
            res = detector.detect_odor()
            print(res)
    except Exception as e:
        print(f"\n❌ 오류: {e}")

if __name__ == '__main__':
    main()