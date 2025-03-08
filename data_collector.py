"""
Optimized Data Collection Module for IoT Huffman Compression
"""

import pandas as pd
import numpy as np
from typing import Generator, List, Optional
import time
import logging
from pathlib import Path
import config

logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataCollector:
    """Optimized data collection for IoT sensors and CSV files"""
    
    def __init__(self):
        self.csv_data: Optional[pd.DataFrame] = None
        self._setup_collector()
    
    def _setup_collector(self):
        """Initialize data source"""
        try:
            csv_path = Path(config.CSV_PATH)
            if csv_path.exists():
                # Read only numeric columns for efficiency
                self.csv_data = pd.read_csv(csv_path, usecols=['temperature', 'humidity', 'voltage'])
                logging.info(f"Loaded CSV data from {config.CSV_PATH}")
            else:
                logging.warning(f"CSV file not found: {config.CSV_PATH}")
        except Exception as e:
            logging.error(f"Failed to load CSV data: {str(e)}")
            self.csv_data = None
    
    def read_csv_data(self, chunk_size: int = config.BATCH_SIZE) -> Generator[List[float], None, None]:
        """Read and preprocess data from CSV file in optimized chunks"""
        if self.csv_data is None:
            raise RuntimeError("CSV data not loaded")
        
        try:
            total_rows = len(self.csv_data)
            
            # Pre-process data for better compression
            # Normalize data to reduce value ranges
            temp_mean = self.csv_data['temperature'].mean()
            temp_std = self.csv_data['temperature'].std()
            hum_mean = self.csv_data['humidity'].mean()
            hum_std = self.csv_data['humidity'].std()
            volt_mean = self.csv_data['voltage'].mean()
            volt_std = self.csv_data['voltage'].std()
            
            normalized_data = pd.DataFrame({
                'temperature': (self.csv_data['temperature'] - temp_mean) / temp_std,
                'humidity': (self.csv_data['humidity'] - hum_mean) / hum_std,
                'voltage': (self.csv_data['voltage'] - volt_mean) / volt_std
            })
            
            # Store normalization parameters for later use
            self.normalization_params = {
                'temp_mean': temp_mean, 'temp_std': temp_std,
                'hum_mean': hum_mean, 'hum_std': hum_std,
                'volt_mean': volt_mean, 'volt_std': volt_std
            }
            
            # Process data in memory-efficient chunks
            for i in range(0, total_rows, chunk_size):
                chunk = normalized_data.iloc[i:i+chunk_size]
                
                # Convert numeric values to list with efficient vectorization
                data = []
                # Use numpy's efficient array operations
                temp_values = chunk['temperature'].to_numpy()
                hum_values = chunk['humidity'].to_numpy()
                volt_values = chunk['voltage'].to_numpy()
                
                # Interleave values efficiently
                for t, h, v in np.column_stack((temp_values, hum_values, volt_values)):
                    data.extend([t, h, v])
                
                yield data
                
                # Log progress for large files
                if total_rows > 1000 and i % 1000 == 0:
                    logging.info(f"Processed {i}/{total_rows} rows from CSV")
                    
        except Exception as e:
            logging.error(f"Error reading CSV data: {str(e)}")
            raise
    
    def denormalize_data(self, data: List[float]) -> List[float]:
        """Denormalize data back to original scale"""
        if not hasattr(self, 'normalization_params'):
            return data
        
        denormalized = []
        for i in range(0, len(data), 3):
            if i + 2 < len(data):
                # Temperature
                temp = data[i] * self.normalization_params['temp_std'] + self.normalization_params['temp_mean']
                # Humidity
                hum = data[i+1] * self.normalization_params['hum_std'] + self.normalization_params['hum_mean']
                # Voltage
                volt = data[i+2] * self.normalization_params['volt_std'] + self.normalization_params['volt_mean']
                denormalized.extend([temp, hum, volt])
        
        return denormalized
    
    def generate_test_data(self, size: int = 1000) -> List[float]:
        """Generate realistic IoT sensor data patterns for testing"""
        time_points = np.linspace(0, 24, size)  # 24-hour pattern
        
        # Temperature pattern (23-25°C with daily variation)
        base_temp = 24.0
        daily_variation = 1.0 * np.sin(2 * np.pi * time_points / 24)
        noise = np.random.normal(0, 0.1, size)  # Reduced noise for more realistic patterns
        
        # Add sudden changes to test pattern recognition
        pattern_changes = np.zeros(size)
        change_points = np.random.choice(size, size=5, replace=False)
        pattern_changes[change_points] = np.random.normal(0, 0.5, 5)
        
        return list(base_temp + daily_variation + noise + pattern_changes) 