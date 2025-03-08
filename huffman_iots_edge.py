import heapq
from collections import Counter, defaultdict
import numpy as np
import math
import sys  # Added sys module import
from typing import List, Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompressionError(Exception):
    """Custom exception for compression-related errors"""
    pass

# Project Context:
# This code implements an advanced version of the Huffman Deep Compression (HDC) algorithm from the IEEE Access paper
# "Huffman Deep Compression of Edge Node Data for Reducing IoT Network Traffic" (Nasif et al., 2024). The goal is to
# compress sensor data at IoT edge nodes (e.g., in smart cities) to reduce network traffic while respecting memory
# constraints (e.g., 80 KB). This version enhances the original with adaptive windows, hybrid compression, and neural-inspired techniques.

# Simulated sensor data (numeric time-series, e.g., temperature readings from an air pollution sensor)
data = [23, 24, 25, 24, 23, 22, 23, 24, 25, 26, 25, 24, 23, 22, 21, 22, 23]
BASE_WINDOW_SIZE = 5  # Starting window size for adaptive segmentation
MAX_MEMORY = 80 * 1024  # 80 KB in bytes, mimicking IoT edge node memory limits

def validate_input_data(data: List[float]) -> None:
    """Validate input data before processing"""
    if not data:
        raise ValueError("Input data cannot be empty")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All elements must be numeric")

def delta_encode(data: List[float]) -> List[float]:
    """
    Encode data using delta encoding with input validation
    
    Args:
        data: List of numeric values
    Returns:
        List of delta encoded values
    Raises:
        ValueError: If input data is invalid
    """
    validate_input_data(data)
    try:
        encoded = [data[0]]
        for i in range(1, len(data)):
            encoded.append(data[i] - data[i-1])
        return encoded
    except Exception as e:
        logger.error(f"Delta encoding failed: {str(e)}")
        raise CompressionError(f"Failed to perform delta encoding: {str(e)}")

def calculate_entropy(segment: List[float]) -> float:
    """
    Calculate entropy with error handling and validation
    
    Args:
        segment: List of numeric values
    Returns:
        float: Entropy value
    Raises:
        ValueError: If segment is invalid
    """
    if not segment:
        return 0.0
    try:
        counts = Counter(segment)
        total = len(segment)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        return entropy
    except Exception as e:
        logger.error(f"Entropy calculation failed: {str(e)}")
        return float('inf')  # Return maximum entropy on error

def adaptive_sliding_window(data: List[float], base_size: int) -> List[List[float]]:
    """
    Perform adaptive sliding window segmentation with memory monitoring
    
    Args:
        data: Input data list
        base_size: Initial window size
    Returns:
        List of segmented data
    Raises:
        CompressionError: If memory limit is exceeded
    """
    validate_input_data(data)
    if base_size <= 0:
        raise ValueError("Base size must be positive")
    
    segments = []
    i = 0
    memory_usage = 0
    
    try:
        while i < len(data):
            # Monitor memory usage
            current_memory = sys.getsizeof(segments)
            if current_memory > MAX_MEMORY:
                logger.warning("Memory limit approaching, adjusting window size")
                base_size = max(2, base_size // 2)  # Reduce window size
            
            test_segment = data[i:i + base_size]
            entropy = calculate_entropy(test_segment)
            window_size = min(int(base_size * (2 - entropy)), len(data) - i)
            segments.append(data[i:i + window_size])
            i += window_size
            
            # Log progress for long sequences
            if len(data) > 1000 and i % 1000 == 0:
                logger.info(f"Processed {i}/{len(data)} elements")
                
        return segments
    except Exception as e:
        logger.error(f"Sliding window processing failed: {str(e)}")
        raise CompressionError(f"Failed to process sliding window: {str(e)}")

# Step 4: Context-Aware Pattern Matching
# Purpose: Identifies repeating patterns in sensor data, weighting them by length and redundancy to suit IoT data characteristics.
def get_patterns(segment):
    patterns = defaultdict(int)
    # Limit pattern length to 4 to keep memory usage low, focusing on short, frequent sensor data sequences
    for size in range(1, min(4, len(segment) + 1)):
        for i in range(len(segment) - size + 1):
            pattern = tuple(segment[i:i + size])  # Use tuple for hashable keys
            # Weight reflects pattern importance: size * redundancy factor
            patterns[pattern] += size * (segment.count(pattern[0]) if size == 1 else 1)
    return patterns

# Step 5: Build Huffman Tree
# Purpose: Constructs a Huffman tree for each segment based on pattern weights, a core component of HDC for lossless compression.
def build_huffman_tree(patterns):
    heap = [[weight, [pattern, ""]] for pattern, weight in patterns.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]  # Left branch
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]  # Right branch
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# Step 6: Neural-Inspired Pruning/Pooling
# Purpose: Merges Huffman trees across segments, pruning redundant patterns and pooling shortest codes using a simple ML-like approach.
def neural_prune_pool(trees, max_codes=10):
    # Sub-function: Predicts pattern importance using a perceptron-like model (mimics deep learning from the paper)
    def predict_importance(pattern, weight):
        features = np.array([len(pattern), weight, 1.0])  # Length, weight, constant
        weights = np.array([0.3, 0.5, 0.2])  # Heuristic weights for scoring
        score = np.dot(features, weights)
        return score

    codebook = {}
    all_patterns = {}
    for tree in trees:
        for pattern, code in tree:
            all_patterns[pattern] = all_patterns.get(pattern, 0) + 1

    # Score patterns and prune to top N (e.g., 10) to fit IoT memory
    scored_patterns = [(pattern, predict_importance(pattern, weight))
                       for pattern, weight in all_patterns.items()]
    scored_patterns.sort(key=lambda x: x[1], reverse=True)
    top_patterns = [p[0] for p in scored_patterns[:max_codes]]

    # Pool: Keep shortest code for each top pattern
    for tree in trees:
        for pattern, code in tree:
            if pattern in top_patterns and (pattern not in codebook or len(code) < len(codebook[pattern])):
                codebook[pattern] = code
    return codebook

# Step 7: Encode Data (Batched for Energy Efficiency)
# Purpose: Compresses data using the final codebook in batches, simulating energy-efficient operation for IoT nodes.
def encode_data(data, codebook, batch_size=5):
    encoded = ""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        j = 0
        while j < len(batch):
            for pattern in sorted(codebook.keys(), key=len, reverse=True):
                if j + len(pattern) <= len(batch) and tuple(batch[j:j + len(pattern)]) == pattern:
                    encoded += codebook[pattern]
                    j += len(pattern)
                    break
            else:
                encoded += str(batch[j])  # Fallback if no pattern matches
                j += 1
    return encoded

# Main Execution
# Project Goal: Demonstrate HDC's ability to compress IoT sensor data efficiently on a Windows machine,
# with enhancements beyond the original paper for better performance.

# Preprocess with Delta Encoding
delta_data = delta_encode(data)
print(f"Original Data: {data}")
print(f"Delta Encoded: {delta_data}")

# Adaptive Window Segmentation
segments = adaptive_sliding_window(delta_data, BASE_WINDOW_SIZE)
print(f"Segments: {segments}")

# Build Huffman Trees
all_trees = []
total_memory = 0
for segment in segments:
    patterns = get_patterns(segment)
    tree = build_huffman_tree(patterns)
    all_trees.append(tree)
    # Estimate memory usage to ensure it fits within IoT limits
    total_memory += sum(sys.getsizeof(p) + sys.getsizeof(c) for p, c in tree)
    if total_memory > MAX_MEMORY:
        print("Warning: Memory exceeded IoT limit!")
        break

# Prune and Pool
codebook = neural_prune_pool(all_trees)
print(f"Codebook: {codebook}")

# Compress
compressed = encode_data(delta_data, codebook)
print(f"Compressed: {compressed}")

# Calculate Sizes and Compression Ratio
# Note: Original size assumes 32-bit integers (typical for sensor data); compressed size is in bits.
original_size = len(data) * 32
compressed_size = len(compressed)
print(f"Original Size: {original_size} bits")
print(f"Compressed Size: {compressed_size} bits")
print(f"Compression Ratio: {original_size / compressed_size:.2f}")

# Further Refinement Suggestions (Comments)
"""
# Real Sensor Data Integration:
# - Replace the sample 'data' with actual sensor readings from a CSV file.
# - Example:
#   import pandas as pd
#   df = pd.read_csv('sensor_data.csv')
#   data = df['temperature'].tolist()  # Adjust column name as needed
# - Preprocess (e.g., normalize or filter) if required before delta encoding.

# Memory Optimization:
# - Install 'psutil' (pip install psutil) to monitor real memory usage.
# - Example:
#   import psutil
#   process = psutil.Process()
#   memory_usage = process.memory_info().rss  # In bytes
#   if memory_usage > MAX_MEMORY:
#       print(f"Memory usage: {memory_usage / 1024} KB - Exceeds limit!")
# - Adjust window size or prune more aggressively if memory exceeds limit.

# Full Neural Model Integration:
# - Use scikit-learn for a lightweight ML model to enhance pruning/pooling.
# - Example:
#   from sklearn.linear_model import LogisticRegression
#   X = [[len(p), w, 1] for p, w in all_patterns.items()]  # Features
#   y = [1 if w > median_weight else 0]  # Binary target (keep/prune)
#   model = LogisticRegression().fit(X, y)
#   scores = model.predict_proba(X)[:, 1]  # Probability to keep
#   top_patterns = [p for p, s in zip(all_patterns.keys(), scores) if s > 0.5]
# - Train offline and load pre-trained weights to keep it lightweight.
"""