"""냄새 농도 예측 모델"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from typing import Dict, List


class ConcentrationPredictor:
    """냄새 농도 예측 모델 클래스"""
    
    def __init__(self, model_type: str = 'svr', **kwargs):
        """
        회귀 모델 초기화
        
        Args:
            model_type: 모델 종류 ('svr', 'linear')
            **kwargs: 모델 하이퍼파라미터
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._create_model(model_type, **kwargs)
        self.is_trained = False
    
    def _create_model(self, model_type: str, **kwargs):
        """
        모델 생성
        
        Args:
            model_type: 모델 종류
            **kwargs: 하이퍼파라미터
        
        Returns:
            생성된 모델
        """
        if model_type == 'svr':
            return SVR(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 100),
                epsilon=kwargs.get('epsilon', 0.1),
                gamma='scale',
            )
        elif model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              cv: int = 5) -> Dict[str, float]:
        """
        모델 훈련
        
        Args:
            X_train: 훈련 특징
            y_train: 훈련 라벨 (농도)
            cv: 교차 검증 폴드 수
        
        Returns:
            평가 지표 딕셔너리
        """
        # 데이터 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 모델 훈련
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # 교차 검증 (R² 점수)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=cv, scoring='r2')
        
        # 훈련 점수
        train_score = self.model.score(X_train_scaled, y_train)
        
        metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_r2': train_score,
        }
        
        print(f"농도 예측 모델 훈련 완료:")
        print(f"  - 교차 검증 R² 점수: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"  - 훈련 R² 점수: {metrics['train_r2']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        냄새 농도 예측
        
        Args:
            X: 입력 특징
        
        Returns:
            예측된 농도
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filepath: Path) -> None:
        """
        모델 저장
        
        Args:
            filepath: 저장 경로
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
        }, filepath)
        print(f"농도 예측 모델 저장 완료: {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'ConcentrationPredictor':
        """
        저장된 모델 로드
        
        Args:
            filepath: 모델 파일 경로
        
        Returns:
            로드된 모델 객체
        """
        data = joblib.load(filepath)
        
        predictor = ConcentrationPredictor(model_type=data['model_type'])
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.is_trained = True
        
        print(f"농도 예측 모델 로드 완료: {filepath}")
        return predictor
