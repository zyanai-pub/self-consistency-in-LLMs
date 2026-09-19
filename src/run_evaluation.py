import json
import os
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
import asyncio
import time

from src import metrics_calculator
from src.evaluation_module.consensus import ConsensusManager
from src.controller.framework_controller import FrameworkController
from src.input_layer.benchmark_loader import BenchmarkLoader
from src.evaluation_module.extractor import AnswerExtractor
from src.models.model_manager import ModelManager

load_dotenv()

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "local")

if EXECUTION_MODE == "local":
    MODELS = {
        "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    }
    # Smaller, faster model for SeerSC budget estimation
    SYSTEM1_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    # Define the two distinct local ports
    SYS2_API_BASE = "http://localhost:8000/v1"
    SYS1_API_BASE = "http://localhost:8001/v1"

elif EXECUTION_MODE == "hybrid":
    MODELS = {
        "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    }
    SYSTEM1_MODEL = "groq/llama-3.1-8b-instant"

    SYS2_API_BASE = "http://localhost:8000/v1"
    SYS1_API_BASE = None  # Groq uses default cloud routing

else:  # "remote"
    MODELS = {
        "groq-llama": "groq/llama-3.3-70b-versatile",
    }
    SYSTEM1_MODEL = "groq/llama-3.1-8b-instant"

    SYS2_API_BASE = None
    SYS1_API_BASE = None

STRATEGIES = ["baseline", "esc", "seer", "ralu"]

SUBSET_SIZE = 200
SUBSET_SEED = 7

STRATEGY_KWARGS = {
    "baseline": {"num_paths": 10, "temperature": 0.7, "top_p": 0.95},
    "esc": {"max_paths": 15, "batch_size": 3, "entropy_threshold": 0.5, "temperature": 0.7, "top_p": 0.95},
    "seer": {"m": 5, "n": 15, "temperature": 0.7, "top_p": 0.95},
    "ralu": {"num_paths": 5, "temperature": 0.7, "top_p": 0.95}
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def build_controller(
        model_name: str,
        api_keys: Dict[str, str],
        sys2_base: str = None,
        sys1_base: str = None
) -> FrameworkController:
    model_manager = ModelManager(model_name, api_keys, api_base=sys2_base)

    system1_model_manager = ModelManager(SYSTEM1_MODEL, api_keys, api_base=sys1_base)

    extractor = AnswerExtractor()
    consensus_manager = ConsensusManager()

    return FrameworkController(
        model_manager=model_manager,
        extractor=extractor,
        consensus_builder=consensus_manager,
        system1_model_manager=system1_model_manager
    )


async def evaluate_single_sample(i: int, item: dict, controller: FrameworkController, strat: str, kwargs: dict) -> dict:
    question = item["question"]
    expected = item["answer"]

    try:
        output = await asyncio.to_thread(controller.execute_task, question, strat, **kwargs)
        prediction = output.get("answer")
        is_correct = AnswerExtractor.answers_are_equal(expected, prediction)

        return {
            "i": i,
            "question": question,
            "prediction": prediction,
            "expected": expected,
            "correct": is_correct,
            "paths_sampled": output.get("paths_sampled"),
            "time_seconds": output.get("time_seconds"),
            "entropy": output.get("entropy", output.get("system1_entropy"))
        }

    except Exception as e:
        print(f"ERROR on item {i}: {e}")
        return {
            "i": i,
            "question": question,
            "expected": expected,
            "prediction": None,
            "correct": False,
            "error": str(e)
        }


async def run_evaluation_async(api_keys: Dict[str, str], subset_size: int = SUBSET_SIZE, subset_seed: int = SUBSET_SEED,
                               models: Dict[str, str] = None, strategies: List[str] = None) -> Dict[str, Any]:
    strategies = strategies or STRATEGIES
    models = models or MODELS

    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_res = {}
    metrics_rows = []

    print("Loading GSM8K subset...")
    loader = BenchmarkLoader()
    subset = loader.get_random_subset(subset_size, subset_seed)
    print(f"Loaded {len(subset)} samples\n")

    for model_label, model_name in models.items():
        print(f"Model: {model_label} ({model_name})\n")

        all_res[model_label] = {}
        controller = build_controller(
            model_name,
            api_keys,
            sys2_base=SYS2_API_BASE,
            sys1_base=SYS1_API_BASE
        )

        for strat in strategies:
            print(f"Strategy: {strat}")
            kwargs = STRATEGY_KWARGS[strat]

            sys2, sys1 = controller.model_manager, controller.system1_model_manager
            sys2.reset_usage()
            sys1.reset_usage()
            t0 = time.perf_counter()

            tasks = [evaluate_single_sample(i, item, controller, strat, kwargs)
                     for i, item in enumerate(subset)]
            strat_res = sorted(await asyncio.gather(*tasks), key=lambda x: x["i"])

            wall = time.perf_counter() - t0
            m = metrics_calculator.compute_metrics(model_label, strat, strat_res, sys2, sys1, wall)
            metrics_rows.append(m)
            print(f"acc={m['accuracy']}  tokens/sample={m['total_tokens_per_sample']}\n")

            all_res[model_label][strat] = {**m, "results": strat_res}

            checkpoint_path = os.path.join(
                RESULTS_DIR,
                f"{run_id}_{model_label}_{strat}.json"
            )
            with open(checkpoint_path, "w") as f:
                json.dump(all_res[model_label][strat], f)

    metrics_calculator.print_metrics_table(metrics_rows)
    metrics_calculator.write_metrics(metrics_rows, os.path.join(RESULTS_DIR, f"{run_id}_summary_metrics"))
    print(f"Saved metrics at {RESULTS_DIR}")

    return all_res


def run_evaluation(api_keys: Dict[str, str], subset_size: int = SUBSET_SIZE, subset_seed: int = SUBSET_SEED,
                   models: Dict[str, str] = None, strategies: List[str] = None):
    # Entry point wrapper to run the async loop
    asyncio.run(run_evaluation_async(api_keys, subset_size, subset_seed, models, strategies))


if __name__ == "__main__":
    api_keys = {
        "gemini": os.environ.get("GOOGLE_API_KEY", ""),
        "groq": os.environ.get("GROQ_API_KEY", "")
    }

    run_evaluation(api_keys)