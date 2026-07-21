import math
from typing import Dict, Any, Optional
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
        if "huggingface" in api_keys:
            os.environ["HUGGINGFACE_API_KEY"] = api_keys["huggingface"]

        # Disable paid add-ons
        litellm.telemetry = False


    def generate_inference(self, prompt: str, max_retries: int=6, **kwargs) -> dict | None:
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
                    top_logprobs=5,
                    **kwargs
                )

                message, confidence, entropy = _get_message_and_scores_from_response(response)

                return {
                    "message": message,
                    "confidence": confidence,
                    "entropy": entropy
                }


            except RateLimitError as e:
                if attempt == max_retries - 1:
                    print(f"max retries reached for {self.model_name}. Skipping prompt")
                    return None

                wait_time_seconds = base_wait_time_seconds * (2**attempt)
                print(f"Retrying in {wait_time_seconds}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time_seconds)

            except BadRequestError as e:
                print(f"Invalid context or param for {self.model_name}: {e}")
                return None
        return None


def _get_message_and_scores_from_response(response) -> tuple[Any, float, float]:
    choices = response.choices[0]
    token_logprobs = choices.logprobs.content

    conf_scores = []
    entropy_scores = []

    for token_info in token_logprobs:
        top_logprobs = token_info.top_logprobs

        probs = []
        for prob in top_logprobs:
            probs.append(math.exp(prob.logprob))

        total_prob = sum(probs)
        normalized_probs = [prob / total_prob for prob in probs]

        entropy = -sum(prob * math.log2(prob) for prob in normalized_probs if prob > 0)
        entropy_scores.append(entropy)

        chosen_token_prob = math.exp(token_info.logprob)
        conf_scores.append(chosen_token_prob)

    avg_conf = sum(conf_scores) / len(conf_scores)
    avg_entropy = sum(entropy_scores) / len(entropy_scores)

    return response.choices[0].message, avg_conf, avg_entropy