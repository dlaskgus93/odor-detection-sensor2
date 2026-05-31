#!/usr/bin/env python3
"""냄새 분류 및 농도 예측 모델 훈련 스크립트"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# 프로젝트 루트로 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.odor_classifier import OdorClassifier
from models.concentration_predictor import ConcentrationPredictor
from src.utils import ensure_directory_exists, save_json, calculate_metrics, calculate_regression_metrics
import config


class ModelTrainer:
    def __init__(self, training_data_path: Path = None):
        self.training_data_path = training_data_path or config.TRAINING_DATA_PATH
        self.model_dir = config.MODEL_DIR
        ensure_directory_exists(self.model_dir)
        
        self.classifier_path = self.model_dir / 'odor_classifier.pkl'
        self.regressor_path = self.model_dir / 'concentration_predictor.pkl'
        self.results_dir = Path(__file__).parent.parent / 'results'
        ensure_directory_exists(self.results_dir)
        
        self.label_encoder = LabelEncoder()
    
    def load_data(self) -> bool:
        try:
            print(f"\n[데이터 로드 중...] {self.training_data_path}")
            if not self.training_data_path.exists():
                print(f"⚠️ 데이터를 찾을 수 없습니다: {self.training_data_path}")
                return False
            
            data = pd.read_csv(self.training_data_path)
            
            # --- 결측치 완벽 차단 안전장치 ---
            original_len = len(data)
            data = data.dropna()
            if len(data) < original_len:
                print(f"⚠️ 찌꺼기 결측치(NaN)가 발견되어 {original_len - len(data)}개의 데이터가 안전하게 삭제되었습니다.")
            
            if len(data) == 0:
                print("❌ 유효한 데이터가 없습니다. 데이터를 다시 수집해주세요.")
                return False
            # --------------------------------
            
            print(f"✓ 데이터 로드 완료: {len(data)} 건")
            
            label_column = 'odor_label'
            self.y_classifier = data[label_column]
            self.X = data.drop(columns=[label_column])
            self.feature_names = self.X.columns.tolist()
            
            self.y_classifier_encoded = self.label_encoder.fit_transform(self.y_classifier)
            self.y_regressor = (self.X.iloc[:, 0] / self.X.iloc[:, 0].max() * 100).values
            
            return True
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류: {e}")
            return False
    
    def prepare_data(self, test_size: float = 0.2, validation_size: float = 0.1, random_state: int = 42):
        X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
            self.X, self.y_classifier_encoded, self.y_regressor,
            test_size=test_size + validation_size, random_state=random_state, stratify=self.y_classifier_encoded
        )
        
        val_ratio = validation_size / (test_size + validation_size)
        X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
            X_temp, y_class_temp, y_reg_temp,
            test_size=1-val_ratio, random_state=random_state, stratify=y_class_temp
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_class_train, self.y_class_val, self.y_class_test = y_class_train, y_class_val, y_class_test
        self.y_reg_train, self.y_reg_val, self.y_reg_test = y_reg_train, y_reg_val, y_reg_test
        print(f"✓ 데이터 분할 완료 (학습: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)})")
    
    def train_classifier(self, model_type: str = 'random_forest', epochs: int = 100, **kwargs) -> dict:
        print("\n" + "="*70 + "\n🧪 냄새 분류 모델 훈련 중...\n" + "="*70)
        self.classifier = OdorClassifier(
            model_type=model_type,
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 15),
            random_state=42
        )
        train_metrics = self.classifier.train(self.X_train.values, self.y_class_train, cv=5)
        
        y_val_pred = self.classifier.predict(self.X_val.values)
        val_metrics = calculate_metrics(self.y_class_val, y_val_pred)
        
        y_test_pred = self.classifier.predict(self.X_test.values)
        test_metrics = calculate_metrics(self.y_class_test, y_test_pred)
        
        self.classifier.save(self.classifier_path)
        return {'model_type': model_type, 'train': train_metrics, 'validation': val_metrics, 'test': test_metrics, 'y_pred': y_test_pred.tolist()}
    
    def train_regressor(self, model_type: str = 'svr', **kwargs) -> dict:
        print("\n" + "="*70 + "\n📈 농도 예측 모델 훈련 중...\n" + "="*70)
        self.regressor = ConcentrationPredictor(
            model_type=model_type,
            kernel=kwargs.get('kernel', 'rbf'),
            C=kwargs.get('C', 100),
            epsilon=kwargs.get('epsilon', 0.1),
        )
        train_metrics = self.regressor.train(self.X_train.values, self.y_reg_train, cv=5)
        
        y_val_pred = self.regressor.predict(self.X_val.values)
        val_metrics = calculate_regression_metrics(self.y_reg_val, y_val_pred)
        
        y_test_pred = self.regressor.predict(self.X_test.values)
        test_metrics = calculate_regression_metrics(self.y_reg_test, y_test_pred)
        
        self.regressor.save(self.regressor_path)
        return {'model_type': model_type, 'train': train_metrics, 'validation': val_metrics, 'test': test_metrics, 'y_pred': y_test_pred.tolist()}
    
    def generate_classification_report(self, y_true, y_pred, output_path=None):
        label_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=label_names)
        print("\n" + "="*70 + "\n📊 분류 보고서\n" + "="*70)
        print(report)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
    
    def save_results(self, classifier_results: dict, regressor_results: dict):
        results = {
            'timestamp': datetime.now().isoformat(),
            'classifier': classifier_results,
            'regressor': regressor_results,
            'label_names': self.label_encoder.classes_.tolist(),
        }
        results_path = self.results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(results, results_path)

def main():
    parser = argparse.ArgumentParser(description="모델 훈련 스크립트")
    parser.add_argument('--training_data', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--classifier_type', type=str, default='random_forest')
    parser.add_argument('--regressor_type', type=str, default='svr')
    
    args = parser.parse_args()
    
    try:
        trainer = ModelTrainer(training_data_path=Path(args.training_data) if args.training_data else None)
        if not trainer.load_data(): return
        trainer.prepare_data(test_size=args.test_size, validation_size=args.validation_split)
        
        classifier_results = trainer.train_classifier(model_type=args.classifier_type, n_estimators=args.epochs)
        regressor_results = trainer.train_regressor(model_type=args.regressor_type)
        
        trainer.save_results(classifier_results, regressor_results)
        trainer.generate_classification_report(trainer.y_class_test, np.array(classifier_results['y_pred']))
        
        print("\n✅ 모델 훈련이 완벽하게 종료되었습니다!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == '__main__':
    main()