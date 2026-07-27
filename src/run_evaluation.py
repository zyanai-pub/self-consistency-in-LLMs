import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any

from src.evaluation_module.consensus import ConsensusManager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.controller.framework_controller import FrameworkController
from src.input_layer.benchmark_loader import BenchmarkLoader
from evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager

MODELS = {
    "gemini-flash": "gemini/gemini-1.5-flash",
    "groq-llama":   "groq/llama3-8b-8192",
    "groq-mixtral": "groq/mixtral-8x7b-32768",
    "mistral-small": "mistral/mistral-small-latest",
}

SYSTEM1_MODEL = "groq/llama3-8b-8192"

STRATEGIES = {"baseline", "esc", "seer", "ralu"}

SUBSET_SIZE = 200
SUBSET_SEED = 7

STRATEGY_KWARGS = {
    "baseline": {"num_paths": 10},
    "esc": {"max_paths": 15, "batch_size": 3, "entropy_threshold": 0.5},
    "seer": {"m": 5, "n": 15},
    "ralu": {"num_paths": 5}
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def build_controller(model_name: str, api_keys: Dict[str, str]) -> FrameworkController:
    model_manager = ModelManager(model_name, api_keys)
    system1_model_manager = ModelManager(SYSTEM1_MODEL, api_keys)

    extractor = AnswerExtractor()

    consensus_manager = ConsensusManager()

    return FrameworkController(
        model_manager=model_manager,
        extractor=extractor,
        consensus_builder=consensus_manager,
        system1_model_manager=system1_model_manager
    )

def print_summary_table(summary: Dict[str, Any]) -> None:
    print("\nSummary:")
    models = list(summary.keys())

    for model in models:
        print(f"{model}:")

    print()

    strategies = list(next(iter(summary.values())).keys())

    for strategy in strategies:
        print(f"{strategy}:")
        for model in models:
            acc = summary[model][strategy]["accuracy"]
            print(f"{acc}")
        print()

def run_evaluation(api_keys: Dict[str, str], subset_size: int = SUBSET_SIZE, subset_seed: int = SUBSET_SEED, models: Dict[str, str] = None, strategies: List[str] = None) -> Dict[str, Any]:
    strategies = strategies or STRATEGIES
    models = models or MODELS

    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_res = {}

    print("Loading GSM8K subset...")

    loader = BenchmarkLoader()
    subset = loader.get_random_subset(subset_size, subset_seed)
    print(f"Loaded {len(subset)} samples\n")

    for model_label, model_name in models.items():
        print(f"Model: {model_label} ({model_name})\n")

        all_res[model_label] = {}
        controller = build_controller(model_name, api_keys)

        for strat in strategies:
            print(f"Strategy: {strat}")
            kwargs = STRATEGY_KWARGS[strat]
            strat_res = []
            correct = 0

            for i, item in enumerate(subset):
                question = item["question"]
                expected = item["answer"]

                try:
                    output = controller.execute_task(question, strat, **kwargs)
                    prediction = output.get("answer")
                    is_correct = str(prediction).strip() == str(expected).strip()

                    if is_correct:
                        correct += 1

                    strat_res.append({
                        "i": i,
                        "question": question,
                        "prediction": prediction,
                        "expected": expected,
                        "correct": is_correct,
                        "paths_sampled": output.get("paths_sampled"),
                        "time_seconds": output.get("time_seconds")
                    })

                except Exception as e:
                    print(f"ERROR: {e}")
                    strat_res.append({
                        "i": i,
                        "question": question,
                        "expected": expected,
                        "prediction": None,
                        "correct": False,
                        "error": str(e)
                    })

                if (i + 1) % 20 == 0:
                    running_acc = correct / i + 1

                    print(f"{i+1}/{len(subset)} - running accuracy:{running_acc}")

            accuracy = correct / len(subset)
            print(f"Overall accuracy: {accuracy}")

            all_res[model_label][strat] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(subset),
                "results": strat_res
            }

            # Log results in case of crash
            checkpoint_path = os.path.join(
                RESULTS_DIR,
                f"{run_id}_{model_label}_{strat}.json"
            )
            with open(checkpoint_path, "w") as f:
                json.dump(all_res[model_label][strat], f)

        sum_path = os.path.join(RESULTS_DIR, f"{run_id}_summary.json")

        summary = {
            model: {
                strategy: {
                    "accuracy": all_res[model_label][strategy]["accuracy"],
                    "correct": all_res[model_label][strategy]["correct"],
                    "total": all_res[model_label][strategy]["total"],
                }
                for strategy in all_res[model]
            }
            for model in all_res
        }
        with open(sum_path, "w") as f:
            json.dump(summary, f)

            print(f"Saved summary at {sum_path}")
        print_summary_table(summary)

    return all_res

if __name__ == "__main__":
    api_keys = {
        "gemini": os.environ.get("GEMINI_API_KEY", ""),
        "groq": os.environ.get("GROQ_API_KEY", "")
    }

    run_evaluation(api_keys)



