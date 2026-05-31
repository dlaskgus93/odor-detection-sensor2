#!/usr/bin/env python3
"""센서 데이터 수집 및 학습 데이터 생성 스크립트"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트로 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensor_data_collector import SensorDataCollector
from src.data_processor import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.utils import ensure_directory_exists, combine_csv_files
import config


class TrainingDataCollector:
    """학습 데이터 수집 드라이버"""
    
    def __init__(self, config_dict: dict = None):
        self.sensor_config = config_dict or config.SENSOR_CONFIG
        self.preprocessing_config = config.PREPROCESSING_CONFIG
        self.feature_config = config.FEATURE_CONFIG
        
        self.USE_SIMULATION = True  # 시뮬레이션 모드 활성화
        
        self.sensor_collector = SensorDataCollector(self.sensor_config)
        self.data_processor = DataProcessor(self.preprocessing_config)
        self.feature_extractor = FeatureExtractor(self.feature_config)
        
        ensure_directory_exists(config.RAW_DATA_DIR)
        ensure_directory_exists(config.PROCESSED_DATA_DIR)
        ensure_directory_exists(config.DATA_DIR / 'training_data')
    
    def simulate_sensor_data(self, odor_type: str, duration: int, sampling_rate: int = 10) -> pd.DataFrame:
        data_list = []
        num_samples = duration * sampling_rate
        interval = 1.0 / sampling_rate
        start_time = time.time()
        
        odor_profiles = {
            'normal': {'gas_mean': 350, 'gas_std': 20, 'gas_trend': 'flat', 'temp_mean': 25, 'humidity_mean': 50},
            'refrigerator': {'gas_mean': 500, 'gas_std': 80, 'gas_trend': 'oscillate', 'temp_mean': 15, 'humidity_mean': 60},
            'industrial': {'gas_mean': 1200, 'gas_std': 150, 'gas_trend': 'increasing', 'temp_mean': 30, 'humidity_mean': 40},
            'decay': {'gas_mean': 800, 'gas_std': 120, 'gas_trend': 'smooth_increase', 'temp_mean': 28, 'humidity_mean': 70},
            'chemical': {'gas_mean': 1500, 'gas_std': 200, 'gas_trend': 'spike', 'temp_mean': 25, 'humidity_mean': 30},
            'smoke': {'gas_mean': 1000, 'gas_std': 100, 'gas_trend': 'rapid_increase', 'temp_mean': 35, 'humidity_mean': 20},
        }
        profile = odor_profiles.get(odor_type.lower(), odor_profiles['normal'])
        
        for i in tqdm(range(num_samples), desc=f"{odor_type} 데이터 시뮬레이션", unit="sample"):
            if profile['gas_trend'] == 'flat':
                gas_value = np.random.normal(profile['gas_mean'], profile['gas_std'])
            elif profile['gas_trend'] == 'oscillate':
                gas_value = profile['gas_mean'] + profile['gas_std'] * np.sin(2 * np.pi * i / 50) + np.random.normal(0, 10)
            elif profile['gas_trend'] == 'increasing':
                trend = i / num_samples * 300
                gas_value = profile['gas_mean'] + trend + np.random.normal(0, profile['gas_std'])
            elif profile['gas_trend'] == 'smooth_increase':
                trend = (i / num_samples) ** 1.5 * 200
                gas_value = profile['gas_mean'] + trend + np.random.normal(0, profile['gas_std'])
            elif profile['gas_trend'] == 'spike':
                if 0.4 < (i / num_samples) < 0.6:
                    gas_value = profile['gas_mean'] + 500 + np.random.normal(0, profile['gas_std'])
                else:
                    gas_value = np.random.normal(profile['gas_mean'], profile['gas_std'])
            elif profile['gas_trend'] == 'rapid_increase':
                trend = i / num_samples * 400
                gas_value = profile['gas_mean'] + trend + np.random.normal(0, profile['gas_std'])
            else:
                gas_value = np.random.normal(profile['gas_mean'], profile['gas_std'])
            
            gas_value = max(0, gas_value)
            temperature = profile['temp_mean'] + np.random.normal(0, 1)
            humidity = profile['humidity_mean'] + np.random.normal(0, 3)
            
            data_list.append({
                'gas': gas_value,
                'temperature': temperature,
                'humidity': humidity,
                'timestamp': start_time + i * interval,
                'odor_label': odor_type.lower(),
            })
        
        return pd.DataFrame(data_list)
    
    def collect_raw_data(self, odor_type: str, duration: int, output_path: Path = None, sampling_rate: int = 10) -> Path:
        if output_path is None:
            output_path = config.RAW_DATA_DIR / f"{odor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            output_path = Path(output_path)
        
        if self.USE_SIMULATION:
            raw_data = self.simulate_sensor_data(odor_type, duration, sampling_rate)
        else:
            raw_data = self.sensor_collector.collect_data(duration, sampling_rate)
            raw_data['odor_label'] = odor_type.lower()
        
        ensure_directory_exists(output_path.parent)
        raw_data.to_csv(output_path, index=False)
        return output_path
    
    def process_raw_data(self, input_path: Path, output_path: Path = None) -> Path:
        if output_path is None:
            output_path = config.PROCESSED_DATA_DIR / input_path.name.replace('.csv', '_processed.csv')
        else:
            output_path = Path(output_path)
        
        raw_data = pd.read_csv(input_path)
        processed_data = self.data_processor.preprocess_dataframe(raw_data)
        
        ensure_directory_exists(output_path.parent)
        processed_data.to_csv(output_path, index=False)
        return output_path
    
    def extract_features(self, input_path: Path, output_path: Path = None, window_size: int = 50) -> Path:
        if output_path is None:
            output_path = config.PROCESSED_DATA_DIR / input_path.name.replace('_processed.csv', '_features.csv')
        else:
            output_path = Path(output_path)
        
        processed_data = pd.read_csv(input_path)
        features = self.feature_extractor.extract_features_from_dataframe(processed_data, window_size=window_size)
        
        ensure_directory_exists(output_path.parent)
        features.to_csv(output_path, index=False)
        return output_path
    
    def process_all_data(self, raw_dir: Path = None):
        raw_dir = raw_dir or config.RAW_DATA_DIR
        raw_files = list(raw_dir.glob('*.csv'))
        
        if not raw_files:
            return
        
        all_features = []
        for raw_file in raw_files:
            try:
                processed_path = self.process_raw_data(raw_file)
                features_path = self.extract_features(processed_path)
                features_df = pd.read_csv(features_path)
                all_features.append(features_df)
            except Exception as e:
                continue
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            training_path = config.TRAINING_DATA_PATH
            ensure_directory_exists(training_path.parent)
            combined_features.to_csv(training_path, index=False)
            print(f"\n✅ 데이터 저장 성공! 경로: {training_path}")

def main():
    parser = argparse.ArgumentParser(description="냄새 데이터 수집")
    parser.add_argument('--basic', action='store_true')
    parser.add_argument('--odor_type', type=str)
    parser.add_argument('--duration', type=int, default=30)
    args = parser.parse_args()
    
    collector = TrainingDataCollector()
    
    if args.basic:
        print("데이터 수집을 시작합니다...")
        for odor_type in config.ODOR_LABELS.values():
            collector.collect_raw_data(odor_type=odor_type.lower(), duration=args.duration)
        collector.process_all_data()

if __name__ == '__main__':
    main()