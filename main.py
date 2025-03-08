"""
Optimized Main Application for IoT Huffman Compression
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json
import sys
import numpy as np

from data_collector import DataCollector
from huffman_iots_edge import RealTimeCompressor, compress_iot_data, decode_data
from visualizer import CompressionVisualizer
import config

# Setup logging
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CompressionApp:
    """Optimized main application class for IoT data compression"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.compressor = RealTimeCompressor()
        self.visualizer = CompressionVisualizer()
        self.stats: Dict[str, float] = {
            'total_samples': 0,
            'compressed_size': 0,
            'original_size': 0,
            'compression_ratio': 0.0,
            'processing_time': 0.0,
            'memory_usage': 0.0
        }
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        Path('data').mkdir(exist_ok=True)
    
    def process_realtime_data(self, duration_seconds: Optional[int] = None):
        """Process real-time sensor data"""
        logging.info("Starting real-time data processing")
        start_time = time.time()
        compressed_chunks = []
        stats = {
            'total_samples': 0,
            'compressed_size': 0,
            'original_size': 0,
            'compression_ratio': 0.0
        }
        
        try:
            for value in self.collector.read_sensor_data():
                result = self.compressor.process_datapoint(value)
                if result:
                    compressed_chunks.append(result)
                    stats['compressed_size'] += len(result)
                    stats['total_samples'] += 1
                    stats['original_size'] += 32  # Assuming 32-bit float
                
                # Update stats
                if stats['compressed_size'] > 0:
                    stats['compression_ratio'] = (stats['original_size'] / 
                                               stats['compressed_size'])
                
                # Print progress
                if stats['total_samples'] % 100 == 0:
                    self._print_progress(stats)
                
                # Check duration
                if duration_seconds and time.time() - start_time >= duration_seconds:
                    break
            
            # Save results
            self._save_results(compressed_chunks, stats, 'realtime')
            
        except KeyboardInterrupt:
            logging.info("Real-time processing stopped by user")
        except Exception as e:
            logging.error(f"Error in real-time processing: {str(e)}")
            raise
    
    def process_csv_data(self) -> Tuple[float, float]:
        """Process data from CSV file with optimized batch processing"""
        logging.info("Starting CSV data processing")
        start_time = time.time()
        
        try:
            batch_stats = []
            for chunk in self.collector.read_csv_data():
                batch_start = time.time()
                
                # Compress chunk with vectorized operations if enabled
                if config.USE_VECTORIZATION:
                    chunk = np.array(chunk)
                compressed, codebook = compress_iot_data(chunk)
                
                # Update statistics
                self.stats['total_samples'] += len(chunk)
                self.stats['compressed_size'] += len(compressed)
                self.stats['original_size'] += len(chunk) * 32
                batch_time = time.time() - batch_start
                batch_stats.append(batch_time)
                
                # Calculate metrics
                if self.stats['compressed_size'] > 0:
                    self.stats['compression_ratio'] = (
                        self.stats['original_size'] / self.stats['compressed_size']
                    )
                self.stats['processing_time'] = time.time() - start_time
                self.stats['memory_usage'] = (
                    sys.getsizeof(compressed) + sys.getsizeof(codebook)
                ) / 1024  # KB
                
                # Print progress
                self._print_progress()
                
                # Save intermediate results
                self._save_results([compressed], codebook, 'csv')
                
                # Adaptive batch sizing if enabled
                if config.ADAPTIVE_BATCH_SIZING and len(batch_stats) >= 5:
                    self._adjust_batch_size(batch_stats[-5:])
            
            return self.stats['compression_ratio'], self.stats['processing_time']
            
        except Exception as e:
            logging.error(f"Error in CSV processing: {str(e)}")
            raise
    
    def _adjust_batch_size(self, batch_times: List[float]):
        """Dynamically adjust batch size based on processing times"""
        avg_time = np.mean(batch_times)
        if avg_time < 0.1 and config.BATCH_SIZE < 512:
            config.BATCH_SIZE *= 2
            logging.info(f"Increased batch size to {config.BATCH_SIZE}")
        elif avg_time > 0.5 and config.BATCH_SIZE > 128:
            config.BATCH_SIZE //= 2
            logging.info(f"Decreased batch size to {config.BATCH_SIZE}")
    
    def _print_progress(self, stats: dict = None):
        """Print compression progress with detailed metrics"""
        if stats:
            print(
                f"\rProcessed: {stats['total_samples']:,} samples | "
                f"Ratio: {stats['compression_ratio']:.2f}x | "
                f"Memory: {stats['memory_usage']:.1f}KB | "
                f"Time: {stats['processing_time']:.1f}s", 
                end=""
            )
        else:
            print(
                f"\rProcessed: {self.stats['total_samples']:,} samples | "
                f"Ratio: {self.stats['compression_ratio']:.2f}x | "
                f"Memory: {self.stats['memory_usage']:.1f}KB | "
                f"Time: {self.stats['processing_time']:.1f}s", 
                end=""
            )
    
    def _convert_codebook_for_json(self, codebook: Dict) -> Dict:
        """Convert tuple keys to string representation for JSON serialization"""
        return {
            ','.join(map(str, k)) if isinstance(k, tuple) else str(k): v
            for k, v in codebook.items()
        }
    
    def _save_results(self, 
                     compressed_data: List[str],
                     codebook: Optional[dict],
                     source: str):
        """Save compression results with optimized file handling"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save compressed data efficiently
        output_path = Path(config.OUTPUT_DIR) / f"{source}_{timestamp}.bin"
        with open(output_path, 'wb') as f:
            for chunk in compressed_data:
                f.write(chunk.encode())
        
        # Save codebook if available
        if codebook:
            codebook_path = Path(config.OUTPUT_DIR) / f"{source}_{timestamp}_codebook.json"
            with open(codebook_path, 'w') as f:
                json_codebook = self._convert_codebook_for_json(codebook)
                json.dump(json_codebook, f, indent=2)
        
        # Save statistics
        if config.SAVE_STATS:
            stats_path = Path(config.OUTPUT_DIR) / f"{source}_{timestamp}_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
    
    def run_tests(self):
        """Run comprehensive compression tests with optimized handling of large datasets"""
        logging.info("Running compression tests")
        
        test_sizes = [100, 1000, 10000]
        results = []
        
        for size in test_sizes:
            print(f"\nTesting with {size:,} samples...")
            test_data = self.collector.generate_test_data(size)
            
            try:
                # Process data in chunks for large test cases
                chunk_size = min(1000, size)  # Process at most 1000 samples at a time
                compressed_chunks = []
                codebooks = []
                total_original_size = 0
                total_compressed_size = 0
                start_time = time.time()
                
                # Process data in chunks
                for i in range(0, len(test_data), chunk_size):
                    chunk = test_data[i:i+chunk_size]
                    compressed, codebook = compress_iot_data(chunk)
                    compressed_chunks.append(compressed)
                    codebooks.append(codebook)
                    total_original_size += len(chunk) * 32
                    total_compressed_size += len(compressed)
                    
                    # Print progress for large datasets
                    if size > 1000:
                        progress = (i + len(chunk)) / len(test_data) * 100
                        print(f"\rProgress: {progress:.1f}%", end="")
                
                compression_time = time.time() - start_time
                
                # Calculate overall metrics
                compression_ratio = total_original_size / total_compressed_size
                memory_used = sum(sys.getsizeof(c) for c in compressed_chunks) / 1024
                
                results.append({
                    'size': size,
                    'ratio': compression_ratio,
                    'time': compression_time,
                    'memory': memory_used
                })
                
                print(f"\nCompression Ratio: {compression_ratio:.2f}x")
                print(f"Memory Used: {memory_used:.1f}KB")
                print(f"Processing Time: {compression_time:.3f}s")
                
                # Clean up to free memory
                del compressed_chunks
                del codebooks
                
            except Exception as e:
                logging.error(f"Test failed for size {size}: {str(e)}")
                print(f"\nError testing size {size}: {str(e)}")
                continue
        
        # Save test results
        test_path = Path(config.OUTPUT_DIR) / "test_results.json"
        with open(test_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualization report
        print("\nGenerating visualization report...")
        self.visualizer.generate_report()
        print("Report generated in compressed_data/ directory")

def main():
    """Main entry point with enhanced error handling"""
    app = CompressionApp()
    
    try:
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            if command == "csv":
                ratio, time = app.process_csv_data()
                print(f"\n\nFinal Results:")
                print(f"Compression Ratio: {ratio:.2f}x")
                print(f"Total Processing Time: {time:.2f}s")
                print("\nGenerating visualization report...")
                app.visualizer.generate_report()
                print("Report generated in compressed_data/ directory")
            elif command == "test":
                app.run_tests()
            else:
                print("Invalid command. Use: csv or test")
        else:
            print("Usage: python main.py [csv|test]")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 