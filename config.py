"""
Optimized IoT Huffman Compression Configuration Settings
"""

# Data Collection Settings
CSV_ENABLED = True
CSV_PATH = "data/sensor_data.csv"
SAMPLE_RATE = 100  # Hz

# Compression Settings (Optimized based on research)
BASE_WINDOW_SIZE = 16  # Increased for better pattern detection
BATCH_SIZE = 256  # Optimized for 5G packet size
MIN_PATTERN_FREQ = 2  # Minimum frequency for pattern consideration
MAX_PATTERN_LENGTH = 8  # Increased for better compression
COMPRESSION_TARGET = 25.0

# Memory Limits (in KB)
MAX_MEMORY = 80 * 1024  # 80KB for IoT edge nodes

# Feature Weights (Optimized based on empirical testing)
FEATURE_WEIGHTS = {
    'length': 0.40,    # Reduced slightly to prevent overfitting
    'frequency': 0.35, # Maintained for pattern importance
    'variance': 0.15,  # Maintained for pattern diversity
    'entropy': 0.10    # Increased for better adaptation
}

# Neural Pattern Recognition
PATTERN_SCORE_THRESHOLD = 0.6  # Minimum score for pattern acceptance
ADAPTIVE_WINDOW_ENABLED = True
MAX_PATTERNS_PER_WINDOW = 32

# Output Settings
OUTPUT_DIR = "compressed_data"
LOG_FILE = "compression.log"
SAVE_STATS = True

# Processing Settings
BUFFER_SIZE = BATCH_SIZE * 2
PROCESSING_INTERVAL = 0.5  # seconds (reduced for faster processing)

# Optimization Flags
USE_VECTORIZATION = True
ENABLE_PATTERN_CACHE = True
ADAPTIVE_BATCH_SIZING = True 