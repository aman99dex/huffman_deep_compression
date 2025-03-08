"""
Visualization Module for IoT Huffman Compression Results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import numpy as np
from typing import List, Dict
import config
import logging

class CompressionVisualizer:
    """Visualizes compression results and performance metrics"""
    
    def __init__(self):
        self.output_dir = Path(config.OUTPUT_DIR)
        sns.set_style("whitegrid")  # Use seaborn's whitegrid style
    
    def plot_compression_ratio(self, stats_file: Path):
        """Plot compression ratio over time"""
        with open(stats_file) as f:
            stats = json.load(f)
        
        plt.figure(figsize=(10, 6))
        plt.plot(stats['compression_ratio'], 'b-', label='Compression Ratio')
        plt.axhline(y=config.COMPRESSION_TARGET, color='r', linestyle='--', 
                   label='Target Ratio')
        plt.title('Compression Ratio Performance')
        plt.xlabel('Time')
        plt.ylabel('Compression Ratio (x)')
        plt.legend()
        plt.grid(True)
        
        output_path = self.output_dir / 'compression_ratio.png'
        plt.savefig(output_path)
        plt.close()
    
    def plot_memory_usage(self, stats_file: Path):
        """Plot memory usage over time"""
        with open(stats_file) as f:
            stats = json.load(f)
        
        plt.figure(figsize=(10, 6))
        plt.plot(stats['memory_usage'], 'g-', label='Memory Usage')
        plt.axhline(y=config.MAX_MEMORY/1024, color='r', linestyle='--', 
                   label='Memory Limit')
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (KB)')
        plt.legend()
        plt.grid(True)
        
        output_path = self.output_dir / 'memory_usage.png'
        plt.savefig(output_path)
        plt.close()
    
    def plot_sensor_data(self, csv_path: Path):
        """Plot original sensor data patterns"""
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(15, 10))
        
        # Temperature subplot
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['temperature'], 'r-', label='Temperature')
        plt.title('Sensor Data Patterns')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        
        # Humidity subplot
        plt.subplot(3, 1, 2)
        plt.plot(df['timestamp'], df['humidity'], 'b-', label='Humidity')
        plt.ylabel('Humidity (%)')
        plt.legend()
        plt.grid(True)
        
        # Voltage subplot
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['voltage'], 'g-', label='Voltage')
        plt.xlabel('Time')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / 'sensor_patterns.png'
        plt.savefig(output_path)
        plt.close()
    
    def plot_test_results(self, results_file: Path):
        """Plot test results comparison"""
        with open(results_file) as f:
            results = json.load(f)
        
        sizes = [r['size'] for r in results]
        ratios = [r['ratio'] for r in results]
        times = [r['time'] for r in results]
        memories = [r['memory'] for r in results]
        
        plt.figure(figsize=(15, 10))
        
        # Compression ratio subplot
        plt.subplot(2, 2, 1)
        plt.plot(sizes, ratios, 'bo-')
        plt.title('Compression Ratio vs Data Size')
        plt.xlabel('Data Size (samples)')
        plt.ylabel('Compression Ratio (x)')
        plt.grid(True)
        
        # Processing time subplot
        plt.subplot(2, 2, 2)
        plt.plot(sizes, times, 'ro-')
        plt.title('Processing Time vs Data Size')
        plt.xlabel('Data Size (samples)')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        # Memory usage subplot
        plt.subplot(2, 2, 3)
        plt.plot(sizes, memories, 'go-')
        plt.title('Memory Usage vs Data Size')
        plt.xlabel('Data Size (samples)')
        plt.ylabel('Memory (KB)')
        plt.grid(True)
        
        # Efficiency subplot
        efficiency = [r['ratio']/r['time'] for r in results]
        plt.subplot(2, 2, 4)
        plt.plot(sizes, efficiency, 'mo-')
        plt.title('Compression Efficiency vs Data Size')
        plt.xlabel('Data Size (samples)')
        plt.ylabel('Ratio/Time')
        plt.grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / 'test_results.png'
        plt.savefig(output_path)
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive visualization report"""
        try:
            # Plot sensor data patterns
            self.plot_sensor_data(Path(config.CSV_PATH))
            
            # Find latest stats file
            stats_files = list(self.output_dir.glob('*_stats.json'))
            if stats_files:
                latest_stats = max(stats_files, key=lambda p: p.stat().st_mtime)
                self.plot_compression_ratio(latest_stats)
                self.plot_memory_usage(latest_stats)
            
            # Plot test results if available
            test_results = self.output_dir / 'test_results.json'
            if test_results.exists():
                self.plot_test_results(test_results)
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            logging.error(f"Visualization error: {str(e)}") 