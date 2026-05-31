"""센서 데이터 수집 모듈"""

import time
import json
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class SensorDataCollector:
    """센서에서 데이터를 수집하는 클래스"""
    
    def __init__(self, config: Dict):
        """
        센서 데이터 수집기 초기화
        
        Args:
            config: 센서 설정 정보
        """
        self.config = config
        self.gas_sensor_config = config.get('gas_sensor', {})
        self.temp_sensor_config = config.get('temperature_sensor', {})
        self.humidity_sensor_config = config.get('humidity_sensor', {})
        
        # 데이터 버퍼
        self.buffer_size = self.gas_sensor_config.get('buffer_size', 100)
        self.gas_data_buffer = deque(maxlen=self.buffer_size)
        self.temp_data_buffer = deque(maxlen=self.buffer_size)
        self.humidity_data_buffer = deque(maxlen=self.buffer_size)
        self.timestamp_buffer = deque(maxlen=self.buffer_size)
        
        # --- 수정된 부분: 에러 알림 상태를 기억하는 스위치 추가 ---
        self._temp_error_printed = False
        self._humid_error_printed = False
        # -----------------------------------------------------------
        
        # 센서 연결 상태
        self.connected = False
        self._init_sensors()
    
    def _init_sensors(self):
        """센서 초기화 (실제 하드웨어와의 연결)"""
        try:
            # Serial 포트 초기화
            import serial
            port = self.gas_sensor_config.get('port', '/dev/ttyUSB0')
            baudrate = self.gas_sensor_config.get('baudrate', 9600)
            
            self.serial_port = serial.Serial(port, baudrate, timeout=1.0)
            self.connected = True
            print(f"센서 연결 성공: {port}")
        except Exception as e:
            print(f"센서 연결 실패: {e}")
            print("시뮬레이션 모드로 진행합니다.")
            self.connected = False
    
    def read_gas_sensor(self) -> float:
        """가스 센서에서 데이터 읽기"""
        if self.connected:
            try:
                # 실제 센서 데이터 읽기
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    try:
                        return float(line)
                    except ValueError:
                        return 0.0
            except Exception as e:
                print(f"가스 센서 읽기 오류: {e}")
                return 0.0
        else:
            # 시뮬레이션: 정상 상태 + 노이즈
            return np.random.normal(400, 50)
    
    def read_temperature(self) -> float:
        """온도 센서에서 데이터 읽기"""
        try:
            if self.temp_sensor_config.get('type') == 'DHT22':
                import Adafruit_DHT
                pin = self.temp_sensor_config.get('pin', 4)
                humidity, temperature = Adafruit_DHT.read_retry(
                    Adafruit_DHT.DHT22, pin
                )
                return temperature if temperature else 25.0
        except Exception as e:
            # --- 수정된 부분: 딱 한 번만 에러를 출력하고 입을 다물게 함 ---
            if not getattr(self, '_temp_error_printed', False):
                print(f"\n⚠️ 온도 센서 읽기 오류 (이후 알림은 생략됩니다): {e}")
                self._temp_error_printed = True
            # -----------------------------------------------------------
        
        # 시뮬레이션
        return np.random.normal(25, 2)
    
    def read_humidity(self) -> float:
        """습도 센서에서 데이터 읽기"""
        try:
            if self.humidity_sensor_config.get('type') == 'DHT22':
                import Adafruit_DHT
                pin = self.humidity_sensor_config.get('pin', 4)
                humidity, temperature = Adafruit_DHT.read_retry(
                    Adafruit_DHT.DHT22, pin
                )
                return humidity if humidity else 50.0
        except Exception as e:
            # --- 수정된 부분: 딱 한 번만 에러를 출력하고 입을 다물게 함 ---
            if not getattr(self, '_humid_error_printed', False):
                print(f"⚠️ 습도 센서 읽기 오류 (이후 알림은 생략됩니다): {e}\n")
                self._humid_error_printed = True
            # -----------------------------------------------------------
        
        # 시뮬레이션
        return np.random.normal(50, 10)
    
    def read_all_sensors(self) -> Dict[str, float]:
        """모든 센서에서 데이터 읽기"""
        return {
            'gas': self.read_gas_sensor(),
            'temperature': self.read_temperature(),
            'humidity': self.read_humidity(),
            'timestamp': time.time(),
        }
    
    def collect_data(self, duration: float, sampling_rate: int = 10) -> pd.DataFrame:
        """주어진 시간 동안 센서 데이터 수집"""
        data_list = []
        interval = 1.0 / sampling_rate
        start_time = time.time()
        
        print(f"데이터 수집 시작 ({duration}초)...")
        
        while time.time() - start_time < duration:
            sensor_data = self.read_all_sensors()
            data_list.append(sensor_data)
            self.gas_data_buffer.append(sensor_data['gas'])
            self.temp_data_buffer.append(sensor_data['temperature'])
            self.humidity_data_buffer.append(sensor_data['humidity'])
            self.timestamp_buffer.append(sensor_data['timestamp'])
            
            time.sleep(interval)
        
        print(f"데이터 수집 완료 ({len(data_list)} samples)")
        return pd.DataFrame(data_list)
    
    def get_buffer_data(self) -> pd.DataFrame:
        """현재 버퍼에 저장된 데이터 반환"""
        if len(self.gas_data_buffer) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'gas': list(self.gas_data_buffer),
            'temperature': list(self.temp_data_buffer),
            'humidity': list(self.humidity_data_buffer),
            'timestamp': list(self.timestamp_buffer),
        })
    
    def save_data(self, data: pd.DataFrame, filepath: Path, odor_label: str = None):
        """수집한 데이터를 파일로 저장"""
        if odor_label:
            data['odor_label'] = odor_label
        
        # CSV로 저장
        data.to_csv(filepath, index=False)
        print(f"데이터 저장 완료: {filepath}")
        
        # 메타데이터도 함께 저장
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(data),
            'odor_label': odor_label,
            'duration': data['timestamp'].iloc[-1] - data['timestamp'].iloc[0] if len(data) > 1 else 0,
        }
        
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def close(self):
        """센서 연결 종료"""
        if self.connected and hasattr(self, 'serial_port'):
            self.serial_port.close()
            print("센서 연결 종료")