import pytest
from src.evaluation_module.extractor import AnswerExtractor

@pytest.fixture
def extractor():
    return AnswerExtractor()

@pytest.mark.parametrize("raw_text, expected_extraction",
                         [
                             ("The answer is 11.", "11"),
                             ("Conclusion: 3 1/4", "3 1/4"),
                             ("First 5, then 6, so the answer is 22.", "22"),
                             ("The soft whisper of the ocean breeze carried the scent of saltwater", None)
                         ])
def test_extraction(extractor, raw_text, expected_extraction):
    assert extractor.extract_from_text(raw_text) == expected_extraction

@pytest.mark.parametrize("extracted_string, expected_valid",
                         [
                             ("17", "17"),
                             ("17.0", "17"),
                             ("3 1/4", "3.25"),
                             ("1/4", "0.25"),
                             ("NaN", None),
                             ("invalid", None)
                         ])
def test_validate_format(extractor, extracted_string, expected_valid):
    assert extractor.validate_format(extracted_string) == expected_valid