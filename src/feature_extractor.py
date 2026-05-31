"""특징 추출 모듈"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Dict, List, Tuple


class FeatureExtractor:
    """센서 신호에서 특징을 추출하는 클래스"""
    
    def __init__(self, config: Dict):
        """
        특징 추출기 초기화
        
        Args:
            config: 특징 추출 설정
        """
        self.config = config
    
    # ===== 시간 영역 특징 =====
    
    def extract_time_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        시간 영역 특징 추출
        
        Args:
            data: 입력 신호
        
        Returns:
            특징 딕셔너리
        """
        features = {}
        
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['min'] = np.min(data)
        features['max'] = np.max(data)
        features['median'] = np.median(data)
        features['range'] = features['max'] - features['min']
        
        # 통계 특징
        if len(data) > 1:
            features['skewness'] = stats.skew(data)
            features['kurtosis'] = stats.kurtosis(data)
        
        # RMS (Root Mean Square)
        features['rms'] = np.sqrt(np.mean(data**2))
        
        # Peak-to-peak
        features['peak_to_peak'] = np.max(data) - np.min(data)
        
        # 1분산 (Variance)
        features['variance'] = np.var(data)
        
        # 에너지
        features['energy'] = np.sum(data**2)
        
        return features
    
    # ===== 주파수 영역 특징 =====
    
    def extract_frequency_domain_features(self, data: np.ndarray, 
                                         sampling_rate: float = 10.0) -> Dict[str, float]:
        """
        주파수 영역 특징 추출
        
        Args:
            data: 입력 신호
            sampling_rate: 샘플링 레이트 (Hz)
        
        Returns:
            특징 딕셔너리
        """
        features = {}
        
        # FFT 계산
        fft = np.fft.fft(data)
        freq = np.fft.fftfreq(len(data), 1/sampling_rate)
        magnitude = np.abs(fft)
        
        # FFT 기반 특징
        features['fft_mean'] = np.mean(magnitude)
        features['fft_std'] = np.std(magnitude)
        features['fft_max'] = np.max(magnitude)
        
        # 전력 스펙트럼
        power = magnitude**2
        features['power_mean'] = np.mean(power)
        features['power_max'] = np.max(power)
        
        # 스펙트럼 중심 (Spectral Centroid)
        if np.sum(magnitude) > 0:
            features['spectral_centroid'] = np.sum(freq[:len(freq)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        else:
            features['spectral_centroid'] = 0
        
        # 스펙트럼 롤오프 (Spectral Rolloff)
        cumsum = np.cumsum(power)
        rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
        features['spectral_rolloff'] = freq[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        return features
    
    # ===== 시계열 특징 =====
    
    def extract_timeseries_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        시계열 특징 추출
        
        Args:
            data: 입력 신호
        
        Returns:
            특징 딕셔너리
        """
        features = {}
        
        # 자기상관 (Autocorrelation)
        if len(data) > 2:
            acf_result = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            acf_result = acf_result[len(acf_result)//2:]
            acf_result = acf_result / acf_result[0]
            
            features['autocorr_lag1'] = acf_result[1] if len(acf_result) > 1 else 0
            features['autocorr_lag2'] = acf_result[2] if len(acf_result) > 2 else 0
        
        # 변화율 (Rate of Change)
        diff = np.diff(data)
        features['mean_diff'] = np.mean(diff)
        features['std_diff'] = np.std(diff)
        features['max_diff'] = np.max(np.abs(diff))
        
        # 제로 크로싱 (Zero Crossing Rate)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(data - np.mean(data))))) / 2
        features['zero_crossing_rate'] = zero_crossings / len(data)
        
        return features
    
    # ===== 센서 간 상호작용 특징 =====
    
    def extract_cross_sensor_features(self, gas_data: np.ndarray, 
                                     temp_data: np.ndarray, 
                                     humidity_data: np.ndarray) -> Dict[str, float]:
        """
        센서 간 상호작용 특징 추출
        
        Args:
            gas_data: 가스센서 데이터
            temp_data: 온도센서 데이터
            humidity_data: 습도센서 데이터
        
        Returns:
            특징 딕셔너리
        """
        features = {}
        
        # 센서 간 비율
        gas_mean = np.mean(gas_data) + 1e-8
        temp_mean = np.mean(temp_data) + 1e-8
        humidity_mean = np.mean(humidity_data) + 1e-8
        
        features['gas_temp_ratio'] = gas_mean / temp_mean
        features['gas_humidity_ratio'] = gas_mean / humidity_mean
        features['temp_humidity_ratio'] = temp_mean / humidity_mean
        
        # 센서 간 상관계수
        features['gas_temp_correlation'] = np.corrcoef(gas_data, temp_data)[0, 1] if len(gas_data) > 1 else 0
        features['gas_humidity_correlation'] = np.corrcoef(gas_data, humidity_data)[0, 1] if len(gas_data) > 1 else 0
        
        # 상호작용 특징
        features['gas_temp_interaction'] = gas_mean * temp_mean
        features['gas_humidity_interaction'] = gas_mean * humidity_mean
        
        return features
    
    # ===== 종합 특징 추출 =====
    
    def extract_all_features(self, gas_data: np.ndarray, 
                            temp_data: np.ndarray, 
                            humidity_data: np.ndarray,
                            sampling_rate: float = 10.0) -> pd.DataFrame:
        """
        모든 특징 추출
        
        Args:
            gas_data: 가스센서 데이터
            temp_data: 온도센서 데이터
            humidity_data: 습도센서 데이터
            sampling_rate: 샘플링 레이트
        
        Returns:
            특징 DataFrame
        """
        features = {}
        
        # 가스센서 특징
        features.update({f'gas_td_{k}': v for k, v in self.extract_time_domain_features(gas_data).items()})
        features.update({f'gas_fd_{k}': v for k, v in self.extract_frequency_domain_features(gas_data, sampling_rate).items()})
        features.update({f'gas_ts_{k}': v for k, v in self.extract_timeseries_features(gas_data).items()})
        
        # 온도센서 특징
        features.update({f'temp_td_{k}': v for k, v in self.extract_time_domain_features(temp_data).items()})
        features.update({f'temp_ts_{k}': v for k, v in self.extract_timeseries_features(temp_data).items()})
        
        # 습도센서 특징
        features.update({f'humidity_td_{k}': v for k, v in self.extract_time_domain_features(humidity_data).items()})
        features.update({f'humidity_ts_{k}': v for k, v in self.extract_timeseries_features(humidity_data).items()})
        
        # 센서 간 상호작용 특징
        features.update(self.extract_cross_sensor_features(gas_data, temp_data, humidity_data))
        
        return pd.DataFrame([features])
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                       window_size: int = 50) -> pd.DataFrame:
        """
        데이터프레임에서 특징 추출
        
        Args:
            df: 입력 데이터프레임
            window_size: 윈도우 크기
        
        Returns:
            특징 DataFrame
        """
        features_list = []
        
        # 슬라이딩 윈도우로 특징 추출
        for i in range(0, len(df) - window_size + 1, window_size // 2):
            window = df.iloc[i:i+window_size]
            
            features = self.extract_all_features(
                gas_data=window['gas'].values,
                temp_data=window['temperature'].values,
                humidity_data=window['humidity'].values,
            )
            
            # 라벨 추가
            if 'odor_label' in window.columns:
                features['odor_label'] = window['odor_label'].iloc[0]
            
            features_list.append(features)
        
        return pd.concat(features_list, ignore_index=True)
