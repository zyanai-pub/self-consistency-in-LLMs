import math
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.decoding_strategies.decoding_strategy import DecodingStrategy
from src.evaluation_module.consensus import ConsensusManager
from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager


class SeerSC(DecodingStrategy):
    def __init__(self, extractor: AnswerExtractor, system1_model_manager: ModelManager, system2_model_manager: ModelManager, consensus_manager: ConsensusManager):
        super().__init__(system2_model_manager, extractor)
        self.system1_model_manager = system1_model_manager
        self.consensus_manager = consensus_manager

    @staticmethod
    def _compute_confidence_weighted_entropy(paths: List[Dict[str, Any]]) -> float:
        answer_confidence_sums = {}
        for path in paths:
            answer= path.get("extracted_answer")
            if answer:
                if not answer_confidence_sums.get(answer):
                    answer_confidence_sums[path['extracted_answer']] = 0
                answer_confidence_sums[path['extracted_answer']] += path['confidence']

        if not answer_confidence_sums:
            return 0.0

        total_weight = sum(answer_confidence_sums.values())
        normalised = {c: w / total_weight for c, w in answer_confidence_sums.items()}

        return -sum(w * math.log(w) for w in normalised.values() if w > 0)


    @staticmethod
    def _allocate_budget(entropy: float, m: int, n: int) -> int:
        if m <= 0:
            raise ValueError(f"m must be > 0")

        tau1 = (1/10) * math.log(m)
        tau2 = (1/3) * math.log(m)

        if entropy < tau1:
            return 1

        elif entropy < tau2:
            return n // 2

        else:
            return n

    def execute(self, prompt: str, m: int = 5, n: int = 15, **kwargs) -> Dict[str, Any]:
        # Run small system1 calls in parallel
        def _single_system1_call(_):
            inference = self.system1_model_manager.generate_inference(prompt, **kwargs)
            if inference is None:
                return None
            ans = self.extractor.extract_from_text(inference.get('message'))
            return {
                'extracted_answer': ans,
                'confidence': inference.get('confidence'),
                'entropy': inference.get('entropy'),
                'message': inference.get('message')
            }

        system1_paths = []
        with ThreadPoolExecutor(max_workers=m) as executor:
            futures = [executor.submit(_single_system1_call, i) for i in range(m)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    system1_paths.append(result)

        entropy = self._compute_confidence_weighted_entropy(system1_paths)
        budget = self._allocate_budget(entropy, m, n)

        system2_paths = self.generate_paths(prompt, num_samples=budget, **kwargs)

        answer = self.consensus_manager.get_majority_vote(system2_paths)

        return {
            "answer": answer,
            "paths_sampled": budget,
            "system1_entropy": entropy,
        }