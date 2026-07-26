import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from src.decoding_strategies.decoding_strategy import DecodingStrategy
from src.decoding_strategies.ralu_utils import _build_cfg, _extract_logic_units, _INITIAL_PROGRAM_SYSTEM_PROMPT, \
    _ALIGNMENT_SYSTEM_PROMPT, _build_alignment_prompt, _SYNTHESIS_SYSTEM_PROMPT
from src.evaluation_module.consensus import ConsensusManager
from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager


class RaLUSC(DecodingStrategy):
    def __init__(
        self,
        extractor: AnswerExtractor,
        model_manager: ModelManager,
        consensus_manager: ConsensusManager,
        max_iters: int = 3,
    ):
        super().__init__(model_manager, extractor, consensus_manager)
        self.max_iters = max_iters

    """ 
    Citation: Some of the RaLU code is based on https://github.com/acceptallgood/RaLU/blob/main/src/RaLU.py
    The alignment method is different from the way implemented in the paper. This is due to
    the litellm API limitations. A detailed explanation of this is provided in the report.
    """

    def _align_unit(self, unit: Dict[str, Any], prompt: str, verified_units: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        curr = unit
        for i in range(self.max_iters):
            alignment_prompt = _build_alignment_prompt(curr, prompt, verified_units)
            full_alignment_prompt = f"{_ALIGNMENT_SYSTEM_PROMPT}\n\n{alignment_prompt}"

            model_output = self.model_manager.generate_inference(full_alignment_prompt, **kwargs)

            if not model_output:
                break

            message = model_output.get('message', "")

            result = ""
            for c in message:
                if c.isalpha():
                    result += c
            if result.lower() == "ok":
                return curr

            fix_match = re.search(r'<\s*Fix\s*>(.*?)<\s*/\s*Fix\s*>', message, re.DOTALL | re.IGNORECASE)

            if fix_match is None:
                break

            parts = re.split(r"Analysis\s*:", message, flags=re.IGNORECASE)
            curr = {
                "unit_id": curr["unit_id"],
                "code": fix_match.group(1).strip(),
                "nl_description": parts[1].strip() if len(parts) > 1 else ""
            }

        return curr

    def _synthesise_solution(self, aligned_units: List[Dict[str, Any]], prompt: str, **kwargs) -> Dict[str, Any]:
        reasoning_path_str = ["# Reasoning Path"]

        for unit in aligned_units:
            reasoning_path_str.append(f"Unit {unit['unit_id'] + 1}: {unit['code']}")
            if unit.get("nl_description"):
                reasoning_path_str.append(unit['nl_description'])
            reasoning_path_str.append("")

        synthesis_prompt = f"{_SYNTHESIS_SYSTEM_PROMPT}\n\n{'chr(10)'.join(reasoning_path_str)}\n\nquestion: {prompt}"

        res = self.model_manager.generate_inference(synthesis_prompt, **kwargs)
        return res if res else {"message": "", "confidence": 0.0}

    # ------------------------------------------------------------------
    # Single RaLU path
    # ------------------------------------------------------------------

    def _generate_ralu_path(self, prompt: str, **kwargs) -> Dict[str, Any]:
        model_output = self.model_manager.generate_inference(prompt=_INITIAL_PROGRAM_SYSTEM_PROMPT)
        program = model_output['message']

        cfg = _build_cfg(program)

        logic_units = _extract_logic_units(cfg)

        aligned_units = []
        for unit in logic_units:
            aligned_units.append(self._align_unit(unit, prompt, aligned_units, **kwargs))

        synthesised_solution = self._synthesise_solution(aligned_units, prompt, **kwargs)

        extracted_answer = self.extractor.extract_from_text(synthesised_solution.get('message'))

        return {'extracted_answer': extracted_answer,
                'confidence': synthesised_solution.get('confidence'),
                'message': synthesised_solution.get('message')}


    def execute(self, prompt: str, num_paths: int = 10, **kwargs) -> Dict[str, Any]:
        ralu_paths = []

        with ThreadPoolExecutor(max_workers=num_paths) as executor:
            futures = [
                executor.submit(self._generate_ralu_path, prompt, **kwargs)
                for _ in range(num_paths)
            ]
            for future in futures:
                res = future.result()
                if res:
                    ralu_paths.append(res)

            return {
                "answer": self.consensus_manager.get_majority_vote(ralu_paths),
                "paths_sampled": len(ralu_paths)
            }