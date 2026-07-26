import random
import re
from typing import Dict, List, Any

from datasets import load_dataset

class BenchmarkLoader:
    def __init__(self):
        self.dataset = None


    def load_gsm8k(self) -> None:
        self.dataset = load_dataset("gsm8k", "main", split="test")


    def parse_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        question = problem["question"]
        raw_answer = problem["answer"]

        match = re.search(r"####\s*([\d,\-]+)", raw_answer)
        if match:
            answer = match.group(1).replace(",", "").strip()
        else:
            answer = raw_answer.strip()

        return {
            "question": question,
            "answer": answer
        }


    # passing the entire dataset would incur costs for the LLM use, so a limited subset is used
    def get_random_subset(self, size: int = 200, seed: int = 7) -> List[Dict[str, str]]:
        if self.dataset is None:
            self.load_gsm8k()

        total = len(self.dataset)
        rand = random.Random(seed)
        indices = rand.sample(range(total), min(size, total))

        return [self.parse_problem(self.dataset[i]) for i in indices]
