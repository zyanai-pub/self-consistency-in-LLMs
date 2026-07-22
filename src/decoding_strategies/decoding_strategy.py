from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager


class DecodingStrategy(ABC):
    def __init__(self, model_manager: ModelManager, extractor: AnswerExtractor):
        self.model_manager = model_manager
        self.extractor = extractor

    def generate_paths(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        num_samples = kwargs.get("num_samples", 5)
        generated_paths = []

        for _ in range(num_samples):
            inference = self.model_manager.generate_inference(prompt, **kwargs)
            answer = self.extractor.extract_from_text(inference.get('message'))
            generated_paths.append({'extracted_answer': answer,
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