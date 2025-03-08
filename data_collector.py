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
            
            # Process data in memory-efficient chunks
            for i in range(0, total_rows, chunk_size):
                chunk = self.csv_data.iloc[i:i+chunk_size]
                
                # Convert numeric values to list
                data = []
                for _, row in chunk.iterrows():
                    # Add each sensor reading separately
                    data.extend([
                        row['temperature'],
                        row['humidity'],
                        row['voltage']
                    ])
                
                yield data
                
                # Log progress for large files
                if total_rows > 1000 and i % 1000 == 0:
                    logging.info(f"Processed {i}/{total_rows} rows from CSV")
                    
        except Exception as e:
            logging.error(f"Error reading CSV data: {str(e)}")
            raise
    
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