"""냄새 분류 모델"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, List


class OdorClassifier:
    """냄새 분류 모델 클래스"""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        분류 모델 초기화
        
        Args:
            model_type: 모델 종류 ('random_forest', 'gradient_boosting')
            **kwargs: 모델 하이퍼파라미터
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._create_model(model_type, **kwargs)
        self.is_trained = False
        self.classes_ = None
    
    def _create_model(self, model_type: str, **kwargs):
        """
        모델 생성
        
        Args:
            model_type: 모델 종류
            **kwargs: 하이퍼파라미터
        
        Returns:
            생성된 모델
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 15),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              cv: int = 5) -> Dict[str, float]:
        """
        모델 훈련
        
        Args:
            X_train: 훈련 특징
            y_train: 훈련 라벨
            cv: 교차 검증 폴드 수
        
        Returns:
            평가 지표 딕셔너리
        """
        # 데이터 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 모델 훈련
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # 교차 검증
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv)
        
        metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': self.model.score(X_train_scaled, y_train),
        }
        
        print(f"모델 훈련 완료:")
        print(f"  - ��차 검증 정확도: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"  - 훈련 정확도: {metrics['train_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        냄새 종류 예측
        
        Args:
            X: 입력 특징
        
        Returns:
            예측된 라벨
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        냄새 종류 예측 확률
        
        Args:
            X: 입력 특징
        
        Returns:
            각 클래스별 확률
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        특징 중요도 반환
        
        Args:
            feature_names: 특징 이름 리스트
        
        Returns:
            특징 중요도 데이터프레임
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
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
            'classes': self.classes_,
            'model_type': self.model_type,
        }, filepath)
        print(f"모델 저장 완료: {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'OdorClassifier':
        """
        저장된 모델 로드
        
        Args:
            filepath: 모델 파일 경로
        
        Returns:
            로드된 모델 객체
        """
        data = joblib.load(filepath)
        
        classifier = OdorClassifier(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.classes_ = data['classes']
        classifier.is_trained = True
        
        print(f"모델 로드 완료: {filepath}")
        return classifier
