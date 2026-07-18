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


def test_initialization_sets_environment_variables(mock_api_keys):
    ModelManager("test_model", mock_api_keys)
    
    assert os.environ.get("GEMINI_API_KEY") == "fake_gemini_key"
    assert os.environ.get("GROQ_API_KEY") == "fake_groq_key"

@patch("src.models.model_manager.litellm.completion")
def test_generate_inference_success(mock_completion, manager):
    mock_choice = Mock()
    mock_choice.message.content = "answer"

    mock_completion.return_value.choices = [mock_choice]

    result = manager.generate_inference("question")

    assert result == "answer"
    mock_completion.assert_called_once()

@patch("src.models.model_manager.time.sleep")
@patch("src.models.model_manager.litellm.completion")
def test_generate_inference_rate_limit_triggers_sleep(mock_completion, mock_sleep, manager):
    mock_choice = Mock()
    mock_choice.message.content = "success"
    
    success_response = Mock()
    success_response.choices = [mock_choice]
    
    mock_completion.side_effect = [
        RateLimitError(message="limit reached", llm_provider="gemini", model="gemini"),
        success_response
    ]

    result = manager.generate_inference("Hello", max_retries=3)
    
    assert result == "success"
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