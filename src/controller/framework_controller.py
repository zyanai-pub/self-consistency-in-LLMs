from src.evaluation_module.consensus import ConsensusManager
from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager

class FrameworkController:
    def __init__(self, model_manager: ModelManager, extractor: AnswerExtractor, consensus_builder: ConsensusManager):
        self.model_manager = model_manager
        self.extractor = extractor
        self.consensus_builder = consensus_builder

    def execute_task(self, prompt: str, strategy_name: str, **kwargs) -> dict:
        # Start with zero shot prompting
        zero_shot_prompt = f"{prompt}\nLet's think step by step"

        # Get parameters
        temp = kwargs.get("temperature", 0.7)
        max_paths = kwargs.get("max_paths", 10)
        entropy_threshold = kwargs.get("entropy_threshold", 0.5)

        strategy_name = strategy_name.lower().strip()

        match strategy_name:
            case "baseline":
                return self._execute_baseline_sc(
                    prompt=zero_shot_prompt,
                    paths=max_paths,
                    temp=temp,
                )
            case "esc":
                return self._execute_esc(
                    prompt=zero_shot_prompt,
                    paths=max_paths,
                    temp=temp,
                    entropy_threshold=entropy_threshold,
                )
            case "seer":
                return self._execute_seer_sc(
                    prompt=zero_shot_prompt,
                    paths=max_paths,
                    temp=temp,
                    entropy_threshold=entropy_threshold,
                )
            case "ralu":
                return self._execute_ralu(
                    prompt=zero_shot_prompt,
                    paths=max_paths,
                    temp=temp,
                    entropy_threshold=entropy_threshold,
                )
            case _:
                raise ValueError(f"Unknown strategy '{strategy_name}'")


    def _execute_baseline_sc(self, prompt: str, **kwargs) -> dict:
        # TODO: Execute the sample-and-marginalise technique.
        # TODO: Request inference from the model manager.
        # TODO: Pass generated reasoning paths to the evaluation module for majority voting.
        pass

    def _execute_esc(self, prompt: str, **kwargs) -> dict:
        # TODO: Execute the Early-Stopping Self-Consistency mechanism.
        # TODO: Calculate the entropy of the answer distribution dynamically during sampling.
        # TODO: Halt sampling once the entropy falls below the predefined threshold.
        pass

    def _execute_seer_sc(self, prompt: str, **kwargs) -> dict:
        # TODO: Execute the Seer Self-Consistency mechanism.
        # TODO: Trigger a smaller "System 1" model to calculate the confidence-weighted entropy.
        # TODO: Estimate the compute budget based on the "System 1" output.
        # TODO: Execute full reasoning on the "System 2" model using the allocated budget concurrently.
        pass

    def _execute_ralu(self, prompt: str, **kwargs) -> dict:
        # TODO: Execute the Reasoning-as-Logic-Units strategy.
        # TODO: Identify critical reasoning tokens during generation.
        # TODO: Adjust the logit distributions to align natural language with structured logic.
        pass