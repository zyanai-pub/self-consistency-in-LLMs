import csv
import json
import math
from typing import Any, Dict, List, Optional, Sequence

STRATEGY_ORDER = ["baseline", "esc", "seer", "ralu"]

COLUMNS = [
    "model", "strategy", "n", "accuracy", "mean_paths_sampled",
    "completion_tokens_per_sample", "total_tokens_per_sample",
    "total_tokens_per_correct", "req_latency_s_mean",
    "throughput_samples_per_min", "errors",
]


def _p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))], 3)


def compute_metrics(model: str, strategy: str, results: Sequence[Dict[str, Any]],
                    sys2: Any, sys1: Any = None,
                    wall_seconds: float = 0.0) -> Dict[str, Any]:
    n = len(results)
    if not n:
        raise ValueError(f"no results for {model}/{strategy}")

    correct = sum(1 for r in results if r.get("correct"))
    systems = [s for s in (sys2, sys1) if s is not None]

    prompt = sum(s.prompt_tokens for s in systems)
    completion = sum(s.completion_tokens for s in systems)
    total = prompt + completion
    requests = sum(s.requests for s in systems)
    latencies = [x for s in systems for x in s.latencies]

    return {
        "model": model,
        "strategy": strategy,
        "n": n,
        "accuracy": round(correct / n, 4),
        "correct": correct,
        "errors": sum(1 for r in results if r.get("error")),
        "no_answer": sum(1 for r in results if not r.get("prediction")),
        "mean_paths_sampled": round(sum(r.get("paths_sampled") or 0 for r in results) / n, 2),
        "requests_per_sample": round(requests / n, 2),
        "prompt_tokens_per_sample": round(prompt / n, 1),
        "completion_tokens_per_sample": round(completion / n, 1),
        "total_tokens_per_sample": round(total / n, 1),
        "total_tokens_per_correct": round(total / correct, 1) if correct else None,
        "sys1_tokens": (sys1.prompt_tokens + sys1.completion_tokens) if sys1 else 0,
        "sys2_tokens": sys2.prompt_tokens + sys2.completion_tokens,
        "total_tokens": total,
        "sample_wall_s_mean": round(sum(r.get("time_seconds") or 0.0 for r in results) / n, 3),
        "req_latency_s_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "req_latency_s_p95": _p95(latencies),
        "strategy_wall_min": round(wall_seconds / 60, 2),
        "throughput_samples_per_min": round(n / (wall_seconds / 60), 2) if wall_seconds else None,
        "request_errors": sum(s.errors for s in systems),
    }


def sort_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = {s: i for i, s in enumerate(STRATEGY_ORDER)}
    return sorted(rows, key=lambda r: (r["model"], order.get(r["strategy"], len(order))))


def print_metrics_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str] = COLUMNS) -> None:
    if not rows:
        print("no metrics")
        return
    rows = sort_rows(rows)
    cells = [["-" if r.get(c) is None else str(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(c), max(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    print("  ".join(c.rjust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.rjust(w) for v, w in zip(row, widths)))


def write_metrics(rows: Sequence[Dict[str, Any]], base_path: str) -> None:
    rows = sort_rows(rows)
    with open(f"{base_path}.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(f"{base_path}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)