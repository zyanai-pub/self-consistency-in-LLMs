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
            r"####\s*(-?\d+(?:\.\d+| \d+/\d+|/\d+)?)",
            r"(?:conclusion|therefore).*(?<![\d.])(-?\d+(?:\.\d+| \d+/\d+|/\d+)?)",
            r"(-?\d+(?:\.\d+| \d+/\d+|/\d+)?)",
        ]
        patterns = custom_patterns if custom_patterns else default_patterns
        self.extraction_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


    def extract_from_text(self, raw_text: str) -> str:
        """
        Takes a single reasoning path's raw text and returns the extracted answer.
        """
        if not raw_text:
            return None

        boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", raw_text)
        if boxed_matches:
            inner = boxed_matches[-1].strip()
            for pattern in self.extraction_patterns:
                res = pattern.findall(inner)
                if res:
                    return res[-1].strip()

        lower_text = raw_text.lower()
        split_idx = -1
        keywords = ["####", "answer is", "conclusion", "therefore", "final answer"]

        for keyword in keywords:
            idx = lower_text.rfind(keyword)
            if idx > split_idx:
                split_idx = idx + len(keyword)

        if split_idx != -1:
            target_text = raw_text[split_idx:]
        else:
            lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
            target_text = lines[-1] if lines else raw_text

        for pattern in self.extraction_patterns:
            res = pattern.findall(target_text)
            if res:
                return res[-1].strip()

        return None

    @staticmethod
    def normalise_format(extracted_answer: str) -> str | None:
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
                numerator = float(fraction[0])
                denominator = float(fraction[1])
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
