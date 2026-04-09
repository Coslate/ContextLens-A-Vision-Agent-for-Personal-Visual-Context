
# test feat: add project skeleton with Pydantic schemas and config
python -m pytest tests/test_schemas.py -v

# test feat: add image preprocessing with quality signal extraction
python -m pytest tests/test_preprocess.py -v

# test feat: add EasyOCR wrapper with structured output
python -m pytest tests/test_ocr.py -v

# test feat: add rule-based image type classifier
python -m pytest tests/test_classifier.py -v

# test feat: add confidence calibration engine and failure mode handlers 
python -m pytest tests/test_confidence.py tests/test_failure_handlers.py -v

# test feat: add synthetic test image generator and ground truth annotations
python -m pytest tests/test_generate_images.py -v