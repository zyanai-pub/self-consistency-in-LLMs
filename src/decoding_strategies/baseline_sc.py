from typing import List, Dict, Any

from src.decoding_strategies.decoding_strategy import DecodingStrategy
from src.evaluation_module.consensus import ConsensusManager
from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager

class BaselineSC(DecodingStrategy):
    def __init__(self, extractor: AnswerExtractor, model_manager: ModelManager, consensus_manager: ConsensusManager):
        super().__init__(model_manager, extractor)
        self.consensus_manager = consensus_manager

    def execute(self, prompt: str, **kwargs) -> dict:
        paths = self.generate_paths(prompt, **kwargs)
        answer = self._apply_majority_vote(paths)

        return {
            "answer": answer,
            "paths_sampled": len(paths)
        }


    def _apply_majority_vote(self, paths: List[Dict[str, Any]]) -> str:
        return self.consensus_manager.get_majority_vote(paths)