"""Configuration constants for ContextLens."""

# --- Image Quality Thresholds ---
BLUR_THRESHOLD = 100.0  # Laplacian variance below this → blurry
BRIGHTNESS_LOW = 0.2
BRIGHTNESS_HIGH = 0.9
CONTRAST_LOW = 0.15
ROTATION_ANGLE_THRESHOLD = 15  # degrees

# --- OCR ---
OCR_LANGUAGES = ["en"]
OCR_CONFIDENCE_LOW = 0.4  # below this → ocr_uncertain flag

# --- Type Classifier ---
TYPE_CONFIDENCE_AMBIGUOUS = 0.5  # below this → ambiguous classification

# --- Confidence Calibration ---
QUALITY_PENALTY_BLURRY = 0.5
QUALITY_PENALTY_LOW_BRIGHTNESS = 0.8
QUALITY_PENALTY_LOW_CONTRAST = 0.8
QUALITY_PENALTY_ROTATED = 0.7

EVIDENCE_STRONG_MATCH = 1.0
EVIDENCE_PATTERN_MATCH = 0.95
EVIDENCE_WEAK_MATCH = 0.7
EVIDENCE_MISSING = 0.3

BASELINE_CONFIDENCE = 0.8  # flat confidence for baseline

# --- Cross-Image Linking ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.5
WEIGHT_EMBEDDING = 0.4
WEIGHT_METADATA = 0.6

# metadata overlap bonuses
SAME_MERCHANT_BONUS = 0.3
DATE_PROXIMITY_BONUS = 0.2  # within 3 days
SHARED_PARTICIPANTS_BONUS = 0.3
SHARED_PROJECT_TAGS_BONUS = 0.4
SHARED_TOPICS_BONUS = 0.2
DATE_PROXIMITY_DAYS = 3

# --- Memory Store ---
DEFAULT_DB_PATH = "contextlens_memory.db"

# --- Evaluation ---
FUZZY_MATCH_THRESHOLD = 80  # rapidfuzz score threshold
CALIBRATION_BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
