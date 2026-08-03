import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm.exceptions import BadRequestError, RateLimitError

NO_LOGPROBS_PROVIDERS = {"groq"}

CONFIDENCE_SUFFIX = (
    "\n\nAfter your reasoning, output exactly one line in this format:\n"
    "CONFIDENCE: <number between 0.0 and 1.0>"
)

_CONFIDENCE_LINE_RE = re.compile(r'\n?CONFIDENCE:\s*[0-9]*\.?[0-9]+\s*$', re.IGNORECASE)


class ModelManager:
    def __init__(
        self,
        model_name: str,
        api_keys: Optional[Dict[str, str]] = None,
        api_base: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_base = api_base or os.getenv("VLLM_API_BASE")
        self.is_local = self.api_base is not None or "localhost" in model_name or "127.0.0.1" in model_name

        if api_keys and "gemini" in api_keys:
            os.environ["GEMINI_API_KEY"] = api_keys["gemini"]
        if api_keys and "groq" in api_keys:
            os.environ["GROQ_API_KEY"] = api_keys["groq"]

        litellm.telemetry = False

        provider = model_name.split("/")[0].lower() if "/" in model_name else ""
        self.supports_logprobs = self.is_local or (provider not in NO_LOGPROBS_PROVIDERS)

    @staticmethod
    def _confidence_from_logprobs(choice_logprobs) -> float:
        if not choice_logprobs or getattr(choice_logprobs, 'content', None) is None:
            return 0.5
        token_logprobs = choice_logprobs.content
        conf_scores = [math.exp(getattr(t, 'logprob', 0.0)) for t in token_logprobs]
        return sum(conf_scores) / len(conf_scores) if conf_scores else 0.5

    @staticmethod
    def _confidence_from_text(content: str) -> float:
        match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', content or "")
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5

    @staticmethod
    def _strip_confidence_line(content: str) -> str:
        return _CONFIDENCE_LINE_RE.sub('', content or "").rstrip()

    def _process_choice(self, choice) -> Dict[str, Any]:
        content = getattr(getattr(choice, "message", None), "content", "") or ""

        if self.supports_logprobs and getattr(choice, "logprobs", None):
            confidence = self._confidence_from_logprobs(choice.logprobs)
            clean_message = content
        else:
            confidence = self._confidence_from_text(content)
            clean_message = self._strip_confidence_line(content)

        return {
            "message": clean_message,
            "confidence": confidence,
        }

    def generate_inference(
        self,
        prompt: str,
        max_retries: int = 6,
        n: int = 1,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        base_wait_time_seconds = 15.0

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024

        if not self.supports_logprobs:
            prompt = prompt + CONFIDENCE_SUFFIX

        target_model = self.model_name
        if self.is_local and not target_model.startswith("openai/"):
            target_model = f"openai/{self.model_name}"

        for attempt in range(max_retries):
            try:
                if not self.is_local:
                    time.sleep(2.1)

                completion_args = {
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "num_retries": 1,
                    "n": n,
                    "logprobs": self.supports_logprobs,
                    **kwargs
                }

                if self.api_base:
                    completion_args["api_base"] = self.api_base
                    completion_args["api_key"] = "none"

                response = litellm.completion(**completion_args)

                results = [self._process_choice(choice) for choice in getattr(response, "choices", [])]

                return results[0] if n == 1 else results

            except RateLimitError:
                if attempt == max_retries - 1:
                    print(f"Max retries reached for {self.model_name}. Skipping prompt.")
                    return None
                wait_time_seconds = base_wait_time_seconds * (2 ** attempt)
                print(f"Rate limited. Retrying in {wait_time_seconds}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time_seconds)

            except BadRequestError as e:
                print(f"Invalid context or param for {self.model_name}: {e}")
                return None

        return None