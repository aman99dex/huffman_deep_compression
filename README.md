# IoT Huffman Deep Compression

An advanced IoT sensor data compression system that utilizes Huffman coding and AI-based pattern recognition.

## Features

- 25x+ compression ratio
- Real-time sensor data processing
- CSV file input support
- AI-driven pattern optimization
- Memory-efficient processing (80KB limit)
- 5G-optimized batch processing
- Adaptive batch sizing
- Vectorized operations
- Comprehensive error handling

## System Requirements

- Python 3.8+
- 80KB minimum RAM
- Windows/Linux/MacOS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iot-huffman.git
cd iot-huffman
```

2. Install required packages:
```bash
pip install numpy pandas scikit-learn pyserial
```

## Project Structure

```
iot_huffman/
├── data/                      # Input data directory
│   └── sensor_data.csv       # Sample sensor data
├── compressed_data/          # Output directory for compressed files
├── config.py                # Configuration settings
├── data_collector.py        # Data collection and preprocessing
├── huffman_iots_edge.py     # Core compression algorithm
├── main.py                  # Main application logic
└── requirements.txt         # Package dependencies
```

## File Interconnections

1. **main.py**
   - Central orchestrator that ties everything together
   - Uses `CompressionApp` class to manage the compression workflow
   - Handles command-line interface and user interaction
   - Depends on: data_collector.py, huffman_iots_edge.py, config.py

2. **data_collector.py**
   - Handles data input from CSV files
   - Preprocesses and normalizes sensor data
   - Implements efficient batch processing
   - Depends on: config.py

3. **huffman_iots_edge.py**
   - Implements the core compression algorithm
   - Contains AI-driven pattern recognition
   - Manages memory-efficient Huffman encoding
   - Depends on: config.py

4. **config.py**
   - Central configuration for all components
   - Controls compression parameters
   - Manages memory limits and optimization flags
   - Used by: all other files

## Usage

### 1. Process CSV Data
```bash
python main.py csv
```
This will:
- Read data from data/sensor_data.csv
- Apply compression with optimized settings
- Save results in compressed_data/
- Display real-time progress and final metrics

### 2. Run Tests
```bash
python main.py test
```
This will:
- Run compression tests with different data sizes
- Generate performance metrics
- Save test results in compressed_data/test_results.json

## Output Files

Each compression run generates three files:
1. `{source}_{timestamp}.bin`
   - Contains the compressed binary data
   - Format: Binary stream of Huffman-encoded data

2. `{source}_{timestamp}_codebook.json`
   - Contains the Huffman codebook
   - Format: JSON mapping of patterns to codes

3. `{source}_{timestamp}_stats.json`
   - Contains compression statistics
   - Includes: compression ratio, processing time, memory usage

## Performance Monitoring

The system provides real-time monitoring of:
- Samples processed
- Current compression ratio
- Memory usage
- Processing time
- Batch processing statistics

## Error Handling

- All errors are logged in `compression.log`
- Automatic batch size adjustment
- Memory usage monitoring
- Graceful error recovery

## Technical Details

### Compression Process Flow
1. Data Collection (data_collector.py)
   - Read input data in configurable batches
   - Normalize and preprocess sensor readings
   - Combine multiple sensor features

2. Pattern Recognition (huffman_iots_edge.py)
   - AI-driven pattern detection
   - Adaptive window sizing
   - Neural-weighted pattern scoring

3. Compression (huffman_iots_edge.py)
   - Memory-efficient Huffman tree construction
   - Optimized encoding process
   - 5G-aware packet optimization

4. Results Management (main.py)
   - Progress monitoring
   - Statistics calculation
   - File output handling

### Optimization Features
- Vectorized operations for better performance
- Adaptive batch sizing based on processing times
- Pattern caching for repeated sequences
- Memory-aware processing
- 5G-optimized packet sizes

## Performance Metrics

- Compression Ratio: 25x-30x
- Memory Usage: < 80KB
- Processing Time: O(n log n)
- Latency: < 10ms (5G transmission)

## Troubleshooting

1. Memory Issues
   - Check config.py memory limits
   - Reduce BATCH_SIZE if needed
   - Monitor memory usage in stats

2. Performance Issues
   - Enable USE_VECTORIZATION in config.py
   - Adjust BATCH_SIZE for your system
   - Check compression.log for bottlenecks

3. Data Input Issues
   - Verify CSV file format
   - Check data normalization settings
   - Review error messages in compression.log

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 