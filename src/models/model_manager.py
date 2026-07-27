import math
from typing import Dict, Any
import os
import litellm
from litellm.exceptions import RateLimitError, BadRequestError
import time


class ModelManager:
    def __init__(self, model_name: str, api_keys: Dict[str, str]):
        self.model_name = model_name

        # groq, gemini and huggingface, as they are the ones that offer a free tier for developers.
        if "gemini" in api_keys:
            os.environ["GEMINI_API_KEY"] = api_keys["gemini"]
        if "groq" in api_keys:
            os.environ["GROQ_API_KEY"] = api_keys["groq"]
        if "mistral" in api_keys:
            os.environ["MITRAL_API_KEY"] = api_keys["mistral"]

        # Disable paid add-ons
        litellm.telemetry = False

    @staticmethod
    def _get_message_and_confidence_from_response(response) -> tuple[Any, float]:
        token_logprobs = response.choices[0].logprobs.content

        conf_scores = [
            math.exp(token_info.logprob)
            for token_info in token_logprobs
        ]

        avg_conf = sum(conf_scores) / len(conf_scores)
        return response.choices[0].message, avg_conf

    def generate_inference(self, prompt: str, max_retries: int = 6, **kwargs) -> dict | None:
        """
        Route the prompt to the appropriate API caller based on the initialized model.
        """
        base_wait_time_seconds = 15.0

        # Handle free-tier payload rejection with fallback
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024

        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    num_retries=1,
                    logprobs=True,
                    **kwargs
                )

                message, confidence = self._get_message_and_confidence_from_response(response)

                return {
                    "message":    message,
                    "confidence": confidence,
                }

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