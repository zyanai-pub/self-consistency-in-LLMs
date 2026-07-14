import re

class AnswerExtractor:
    # Parses raw LLM outputs and isolates the final answer from the output.

    def __init__(self, custom_patterns=None):
        """
        A list of commonly used patterns of answers in benchmarks and model outputs
        Sorted from the most straightforward used in formatted benchmarks, to the vaguest that simply
        extracts the last number in the text
        """
        default_patterns = [
            r"####\s*(-?\d+(?:\.\d+)?)",
            r"(?:the\s+)?answer\s+is\s*:?\s*\*?\*?\s*(-?\d+(?:\.\d+)?(?:/\d+)?)",
            r"(?:conclusion|therefore)[^0-9]*(-?\d+(?:\.\d+)?(?:/\d+)?)",
            r"(-?\d+(?:\.\d+)?(?:/\d+)?)(?!.*\d)"
        ]
        patterns = custom_patterns if custom_patterns else default_patterns
        self.extraction_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]


    def extract_from_text(self, raw_text: str) -> str:
        """
        Takes a single reasoning path's raw text and returns the extracted answer.
        """
        # TODO: Implement the extraction logic.
        # TODO: Handle edge cases where the answer cannot be found (return None or raise a specific exception).
        # This will be crucial for handling inputs similar to your 'malformed_output.json' mock data.
        pass

    def validate_format(self, extracted_answer: str) -> bool:
        """
        Checks if the extracted answer matches the expected data type/format 
        (e.g., is it a valid integer/float for math problems?).
        """
        # TODO: Implement validation logic to prevent reasoning hallucinations from 
        # passing as valid structural logic.
        pass