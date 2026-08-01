import json
import math
import re
from typing import Dict, Any
import os
import litellm
from litellm.exceptions import RateLimitError, BadRequestError
import time

NO_LOGPROBS_PROVIDERS = {"groq"}

CONFIDENCE_SUFFIX = (
    "\n\nAfter your reasoning, output exactly one line in this format:\n"
    "CONFIDENCE: <number between 0.0 and 1.0>"
)

_CONFIDENCE_LINE_RE = re.compile(r'\n?CONFIDENCE:\s*[0-9]*\.?[0-9]+\s*$', re.IGNORECASE)

# Cache management to handle rate limit
CACHE_FILE = "llm_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        local_cache = json.load(f)
else:
    local_cache = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(local_cache, f, indent=4)

class ModelManager:
    def __init__(self, model_name: str, api_keys: Dict[str, str]):
        self.model_name = model_name

        # groq and gemini, as they are the ones that offer a free tier for developers.
        if "gemini" in api_keys:
            os.environ["GEMINI_API_KEY"] = api_keys["gemini"]
        if "groq" in api_keys:
            os.environ["GROQ_API_KEY"] = api_keys["groq"]

        # Disable paid add-ons
        litellm.telemetry = False

        provider = model_name.split("/")[0].lower()
        self.supports_logprobs = provider not in NO_LOGPROBS_PROVIDERS

    @staticmethod
    def _confidence_from_logprobs(response) -> float:
        token_logprobs = response.choices[0].logprobs.content
        conf_scores = [math.exp(t.logprob) for t in token_logprobs]
        return sum(conf_scores) / len(conf_scores)

    @staticmethod
    def _confidence_from_text(response) -> float:
        text = response.choices[0].message.content or ""
        match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', text)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5

    def _get_message_and_confidence_from_response(self, response) -> tuple[Any, float]:
        message = response.choices[0].message
        if self.supports_logprobs:
            confidence = self._confidence_from_logprobs(response)
        else:
            confidence = self._confidence_from_text(response)
        return message, confidence


    @staticmethod
    def _strip_confidence_line(message) -> str:
        text = message.content if hasattr(message, 'content') else str(message)
        return _CONFIDENCE_LINE_RE.sub('', text).rstrip()

    def generate_inference(self, prompt: str, max_retries: int = 6, **kwargs) -> dict | None:
        """
        Route the prompt to the appropriate API caller based on the initialized model.
        """
        # Cache management
        effective_prompt = prompt + CONFIDENCE_SUFFIX if not self.supports_logprobs else prompt
        cache_key = f"{self.model_name}_{effective_prompt}"
        if cache_key in local_cache:
            print(f"loaded from cache for {self.model_name}")
            return local_cache[cache_key]


        base_wait_time_seconds = 15.0

        # Handle free-tier payload rejection with fallback
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024

        if not self.supports_logprobs:
            prompt = prompt + CONFIDENCE_SUFFIX

        for attempt in range(max_retries):
            try:
                time.sleep(2.1)
                response = litellm.completion(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    num_retries=1,
                    logprobs=self.supports_logprobs,
                    **kwargs
                )

                message, confidence = self._get_message_and_confidence_from_response(response)

                clean_message = message
                if not self.supports_logprobs:
                    clean_message = self._strip_confidence_line(clean_message)

                result =  {
                    "message":    clean_message,
                    "confidence": confidence,
                }

                local_cache[cache_key] = result
                save_cache()

                return result

            except RateLimitError:
                if attempt == max_retries - 1:
                    print(f"max retries reached for {self.model_name}. Skipping prompt")
                    return None

                wait_time_seconds = base_wait_time_seconds * (2 ** attempt)
                print(f"Retrying in {wait_time_seconds}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time_seconds)

            except BadRequestError as e:
                print(f"Invalid context or param for {self.model_name}: {e}")
                return None

        return None