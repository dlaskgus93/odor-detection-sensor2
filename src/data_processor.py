"""데이터 전처리 모듈"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import signal

class DataProcessor:
    """센서 데이터 전처리 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        if method == 'minmax':
            # NaN을 무시하고 안전하게 최소/최대 계산
            return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-8)
        elif method == 'zscore':
            return (data - np.nanmean(data)) / (np.nanstd(data) + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        z_scores = np.abs((data - np.nanmean(data)) / (np.nanstd(data) + 1e-8))
        return data[z_scores < threshold]
    
    def smooth_data(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        if window_size < 3:
            return data
        kernel = np.ones(window_size) / window_size
        # 평활화 전 결측치 임시 보완
        df_temp = pd.Series(data).interpolate(limit_direction='both').bfill().ffill()
        smoothed = np.convolve(df_temp.values, kernel, mode='same')
        return smoothed
    
    def fill_missing_values(self, data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        df = pd.DataFrame({'value': data})
        if method == 'interpolate':
            # limit_direction='both' 및 bfill/ffill로 가장자리 결측치까지 완벽하게 채움
            df['value'] = df['value'].interpolate(method='linear', limit_direction='both')
            df['value'] = df['value'].bfill().ffill()
        elif method == 'forward_fill':
            df['value'] = df['value'].ffill().bfill()
        return df['value'].values
    
    def apply_filter(self, data: np.ndarray, filter_type: str = 'lowpass', 
                     cutoff_freq: float = 0.1, order: int = 4) -> np.ndarray:
        b, a = signal.butter(order, cutoff_freq, btype=filter_type)
        return signal.filtfilt(b, a, data)
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        sensor_columns = ['gas', 'temperature', 'humidity']
        
        for col in sensor_columns:
            if col not in df_processed.columns:
                continue
            
            data = df_processed[col].values
            
            # 1. 결측치 채우기
            if self.config.get('fill_missing', True):
                data = self.fill_missing_values(data, method=self.config.get('missing_method', 'interpolate'))
            
            # 2. 이상치 제거
            if self.config.get('remove_outliers', True):
                threshold = self.config.get('outlier_threshold', 3.0)
                z_scores = np.abs((data - np.nanmean(data)) / (np.nanstd(data) + 1e-8))
                data[z_scores > threshold] = np.nan
                data = self.fill_missing_values(data, method='interpolate')
            
            # 3. 평활
            if self.config.get('smooth_data', True):
                window_size = self.config.get('smoothing_window', 5)
                data = self.smooth_data(data, window_size)
            
            # 4. 정규화
            if self.config.get('normalize', True):
                method = self.config.get('normalization_method', 'minmax')
                data = self.normalize(data, method)
            
            df_processed[col] = data
        
        return df_processed
    
    def create_sliding_windows(self, data: np.ndarray, window_size: int, step_size: int) -> Tuple[np.ndarray, np.ndarray]:
        windows = []
        indices = []
        for i in range(0, len(data) - window_size + 1, step_size):
            windows.append(data[i:i+window_size])
            indices.append(i)
        return np.array(windows), np.array(indices)
    
    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(random_state)
        indices = np.random.permutation(len(df))
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[indices[:split_idx]], df.iloc[indices[split_idx:]]