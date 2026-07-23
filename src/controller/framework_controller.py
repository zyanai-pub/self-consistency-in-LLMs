from src.decoding_strategies.esc import EarlyStoppingSC
from src.decoding_strategies.seer_sc import SeerSC
from src.evaluation_module.consensus import ConsensusManager
from src.evaluation_module.extractor import AnswerExtractor
from src.decoding_strategies.baseline_sc import BaselineSC
import time

from src.models.model_manager import ModelManager


class FrameworkController:
    def __init__(self,
                 model_manager: ModelManager,
                 extractor: AnswerExtractor,
                 consensus_builder: ConsensusManager,
                 system1_model_manager: ModelManager = None):
        self.extractor = extractor
        self.consensus_builder = consensus_builder
        self.model_manager = model_manager
        self.baseline_sc = BaselineSC(self.extractor, self.model_manager, self.consensus_builder)
        self.esc = EarlyStoppingSC(self.extractor, self.model_manager, self.consensus_builder)
        self.seer_sc = SeerSC(self.extractor, self.model_manager, system1_model_manager, self.consensus_builder)

    def execute_task(self, prompt: str, strategy_name: str, **kwargs) -> dict:
        # Start with zero shot prompting
        zero_shot_prompt = f"{prompt}\nLet's think step by step."

        strategy_name = strategy_name.lower().strip()

        start_time = time.time()
        result = {}

        match strategy_name:
            case "baseline":
                result = self.baseline_sc.execute(zero_shot_prompt, **kwargs)
            case "esc":
                result = self.esc.execute(prompt, **kwargs)
            case "seer":
                result = self.seer_sc.execute(prompt, **kwargs)
            case "ralu":
                result = self._execute_ralu(
                    prompt=zero_shot_prompt,
                    **kwargs
                )
            case _:
                raise ValueError(f"Unknown strategy '{strategy_name}'")

        end_time = time.time()

        result.update({"time_seconds": round(end_time - start_time, 3), "strategy": strategy_name})

        return result

    def _execute_ralu(self, prompt: str, **kwargs) -> dict:
        # TODO: Execute the Reasoning-as-Logic-Units strategy.
        # TODO: Identify critical reasoning tokens during generation.
        # TODO: Adjust the logit distributions to align natural language with structured logic.
        pass