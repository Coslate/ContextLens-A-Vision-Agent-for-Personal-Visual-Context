
# test feat: add project skeleton with Pydantic schemas and config
python -m pytest tests/test_schemas.py -v

# test feat: add image preprocessing with quality signal extraction
python -m pytest tests/test_preprocess.py -v

# test feat: add EasyOCR wrapper with structured output
python -m pytest tests/test_ocr.py -v
