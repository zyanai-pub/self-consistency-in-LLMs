from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.evaluation_module.consensus import ConsensusManager
from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager

ANSWER_SUFFIX = """
Solve the problem step by step.
At the end, output exactly one final line in this format:
#### <final numeric answer>
Do not put any other text after that line.
"""


class DecodingStrategy(ABC):
    def __init__(self, model_manager: ModelManager, extractor: AnswerExtractor, consensus_manager: ConsensusManager):
        self.model_manager = model_manager
        self.extractor = extractor
        self.consensus_manager = consensus_manager

    def generate_paths(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        num_samples = kwargs.pop("num_paths", kwargs.pop("num_samples", 5))

        inferences = self.model_manager.generate_inference(
            f"{prompt}\n\n{ANSWER_SUFFIX}",
            n = num_samples,
            **kwargs
        )

        if not inferences:
            return []

        generated_paths = []
        for inference in inferences:
            ans = self.extractor.extract_from_text(inference.get('message'))
            generated_paths.append({
                'extracted_answer': ans,
                'confidence': inference.get('confidence'),
                'message': inference.get('message')
            })
        return generated_paths

    @abstractmethod
    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Abstract method that all decoding strategies must implement
        """
        pass