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
        for pattern in self.extraction_patterns:
            res = pattern.search(raw_text)
            if res:
                # If we get more than one match, we need to find the last one
                match = res[-1].strip()
                return match
        return None

    def validate_format(self, extracted_answer: str) -> bool:
        """
        Checks if the extracted answer matches the expected data format and match to expected format
        """
        if not extracted_answer:
            return None
        
        try:
            # Handle fractions (for example 3 1/4)
            if '/' in extracted_answer:
                nums = extracted_answer.split()
                whole = 0
                if len(nums) == 2:
                    whole = float(nums.pop(0))
                fraction = nums[0].split('/')
                numerator = fraction[0]
                denominator = fraction[1]
                num_val = whole + (numerator / denominator)
            else:
                num_val = float(extracted_answer)
            
            # Check for impossible values (hallucinations)
            if num_val != num_val or num_val in [float('inf'), float('-inf')]:
                return None
            
            if num_val.is_integer():
                return str(int(num_val))
            else:
                return str(round(num_val, 4))

        except ValueError:
            return None
