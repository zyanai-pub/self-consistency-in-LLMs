import math
from collections import Counter
from typing import Dict, Any, List

from src.decoding_strategies.decoding_strategy import DecodingStrategy
from src.evaluation_module.extractor import AnswerExtractor
from src.evaluation_module.consensus import ConsensusManager
from src.models.model_manager import ModelManager


class EarlyStoppingSC(DecodingStrategy):
    def __init__(self, extractor: AnswerExtractor, model_manager: ModelManager, consensus_manager: ConsensusManager):
        super().__init__(model_manager, extractor)
        self.consensus_manager = consensus_manager

    @staticmethod
    def _calculate_entropy(answers: List[str]) -> float:
        if not answers:
            return 0.0

        num_answers = len(answers)

        answer_counts = Counter(answers)

        entropy = 0.0
        for count in answer_counts.values():
            probability = count / num_answers

            # Shannon entropy formula: -p * log2(p)
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return float(entropy)

    def execute(self, prompt: str, max_paths: int = 15, batch_size: int = 3, entropy_threshold: float = 0.5,
                **kwargs) -> Dict[str, Any]:
        all_paths = []
        paths_sampled = 0
        entropy= None

        while paths_sampled < max_paths:
            batch_size = min(batch_size, max_paths - paths_sampled)
            kwargs.pop("num_paths", None)

            paths = self.generate_paths(prompt, num_samples=batch_size, **kwargs)

            all_paths.extend(paths)
            paths_sampled += len(paths)

            if paths_sampled < batch_size*2:
                continue

            extracted_answers = [path.get('extracted_answer') for path in paths if path.get('extracted_answer')]
            entropy = self._calculate_entropy(extracted_answers)

            if entropy <= entropy_threshold:
                break

        answer = self.consensus_manager.get_majority_vote(all_paths)

        return {
            "answer": answer,
            "paths_sampled": paths_sampled,
            "entropy": entropy
        }