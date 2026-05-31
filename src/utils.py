"""유틸리티 함수"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd


def ensure_directory_exists(directory: Path) -> None:
    """
    디렉토리가 없으면 생성
    
    Args:
        directory: 디렉토리 경로
    """
    directory.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    JSON 파일로 저장
    
    Args:
        data: 저장할 데이터
        filepath: 저장 경로
    """
    ensure_directory_exists(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    JSON 파일 로드
    
    Args:
        filepath: 파일 경로
    
    Returns:
        로드된 데이터
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def load_csv_data(filepath: Path) -> pd.DataFrame:
    """
    CSV 파일 로드
    
    Args:
        filepath: 파일 경로
    
    Returns:
        데이터프레임
    """
    return pd.read_csv(filepath)


def combine_csv_files(directory: Path, pattern: str = '*.csv') -> pd.DataFrame:
    """
    디렉토리의 모든 CSV 파일 합치기
    
    Args:
        directory: 디렉토리 경로
        pattern: 파일 패턴
    
    Returns:
        합쳐진 데이터프레임
    """
    csv_files = list(directory.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return pd.DataFrame()
    
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    분류 성능 지표 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
    
    Returns:
        지표 딕셔너리
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    return metrics


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    회귀 성능 지표 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
    
    Returns:
        지표 딕셔너리
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    return metrics
