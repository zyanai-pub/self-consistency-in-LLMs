from typing import List, Dict, Any

from src.evaluation_module.consensus import ConsensusManager
from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager

class BaselineSC:
    def __init__(self, extractor: AnswerExtractor, model_manager: ModelManager, consensus_manager: ConsensusManager):
        self.model_manager = model_manager
        self.consensus_manager = consensus_manager
        self.extractor = extractor

    def generate_paths(self, prompt: str, num_samples: int, **kwargs) -> List[Dict[str, Any]]:
        generated_paths = []

        for _ in range(num_samples):
            inference = self.model_manager.generate_inference(prompt, **kwargs)
            answer = self.extractor.extract_from_text(inference.get('message'))
            generated_paths.append({'extracted_answer': answer,
                                    'confidence': inference.get('confidence'),
                                    'message': inference.get('message')
                                    })

        return generated_paths



    def apply_majority_vote(self, paths: List[Dict[str, Any]]) -> str:
        return self.consensus_manager.get_majority_vote(paths)