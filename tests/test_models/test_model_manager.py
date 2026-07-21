import os
import pytest
from unittest.mock import patch, Mock
from litellm.exceptions import RateLimitError, BadRequestError
from models.model_manager import ModelManager

@pytest.fixture
def mock_api_keys():
    return {
        "gemini": "fake_gemini_key",
        "groq": "fake_groq_key"
    }

@pytest.fixture
def manager(mock_api_keys):
    return ModelManager(model_name="gemini/gemini-1.5-flash", api_keys=mock_api_keys)


def create_mock_response(message_text="the answer is 6"):
    mock_top_prob_1 = Mock()
    mock_top_prob_1.logprob = -0.1

    mock_top_prob_2 = Mock()
    mock_top_prob_2.logprob = -2.3

    mock_token_info = Mock()
    mock_token_info.logprob = -0.1
    mock_token_info.top_logprobs = [mock_top_prob_1, mock_top_prob_2]

    mock_choice = Mock()
    mock_choice.message = message_text
    mock_choice.logprobs.content = [mock_token_info]

    mock_response = Mock()
    mock_response.choices = [mock_choice]

    return mock_response

def test_initialization_sets_environment_variables(mock_api_keys):
    ModelManager("test_model", mock_api_keys)
    
    assert os.environ.get("GEMINI_API_KEY") == "fake_gemini_key"
    assert os.environ.get("GROQ_API_KEY") == "fake_groq_key"


@patch('src.models.model_manager.litellm.completion')
def test_generate_inference_success(mock_completion):
    mock_completion.return_value = create_mock_response()

    manager = ModelManager("gemini-1.5-flash", {"gemini": "fake_key"})
    result = manager.generate_inference("question")

    assert result is not None
    assert result['message'] == "the answer is 6"
    assert 'entropy' in result
    assert 'confidence' in result


@patch("src.models.model_manager.time.sleep")
@patch("src.models.model_manager.litellm.completion")
def test_generate_inference_rate_limit_triggers_sleep(mock_completion, mock_sleep):
    success_response = create_mock_response(message_text="success")

    mock_completion.side_effect = [
        RateLimitError(message="limit reached", llm_provider="gemini", model="gemini"),
        success_response
    ]

    manager = ModelManager("gemini-1.5-flash", {"gemini": "fake_key"})
    result = manager.generate_inference("Hello", max_retries=3)

    assert result["message"] == "success"
    assert "confidence" in result
    assert "entropy" in result

    assert mock_completion.call_count == 2
    mock_sleep.assert_called_once_with(15.0)

@patch("src.models.model_manager.time.sleep")
@patch("src.models.model_manager.litellm.completion")
def test_generate_inference_max_retries_exceeded(mock_completion, mock_sleep, manager):
    mock_completion.side_effect = RateLimitError(
        message="Too Many Requests", llm_provider="gemini", model="gemini"
    )
    
    # max_retries=2 to keep the test fast
    result = manager.generate_inference("Hello", max_retries=2)
    
    assert result is None
    assert mock_completion.call_count == 2
    assert mock_sleep.call_count == 1

@patch("src.models.model_manager.litellm.completion")
def test_generate_inference_bad_request(mock_completion, manager):
    mock_completion.side_effect = BadRequestError(
        message="Context length exceeded", llm_provider="gemini", model="gemini"
    )
    
    result = manager.generate_inference("Hello")
    
    assert result is None
    assert mock_completion.call_count == 1