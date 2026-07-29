import pytest
from unittest.mock import Mock
from src.controller.framework_controller import FrameworkController


@pytest.fixture
def mock_controller():
    mock_model_manager = Mock()
    mock_extractor = Mock()
    mock_consensus_builder = Mock()

    controller = FrameworkController(
        model_manager=mock_model_manager,
        extractor=mock_extractor,
        consensus_builder=mock_consensus_builder
    )
    return controller


def test_execute_task_baseline_routing(mock_controller, monkeypatch):
    mock_execute = Mock(return_value={"answer": "answer", "paths_sampled": 5})
    monkeypatch.setattr(mock_controller.baseline_sc, "execute", mock_execute)

    result = mock_controller.execute_task(
        prompt="question",
        strategy_name="baseline",
        max_paths=10,
        temp=0.7
    )

    assert result["answer"] == "answer"
    mock_execute.assert_called_once_with(
        "question\nLet's think step by step.",  # positional, no prompt=
        max_paths=10,
        temp=0.7
    )


def test_execute_task_esc_custom_parameters(mock_controller, monkeypatch):
    mock_execute_esc = Mock(return_value={"answer": "answer", "paths_sampled": 5})
    monkeypatch.setattr(mock_controller.esc, "execute", mock_execute_esc)

    mock_controller.execute_task(
        prompt="question",
        strategy_name="esc",
        max_paths=20,
        entropy_threshold=0.2,
        temp=0.5
    )

    mock_execute_esc.assert_called_once_with(
        "question\nLet's think step by step.",  # positional, no prompt=
        max_paths=20,
        entropy_threshold=0.2,
        temp=0.5
    )


def test_execute_task_invalid_strategy(mock_controller):
    with pytest.raises(ValueError, match="Unknown strategy 'strat'"):
        mock_controller.execute_task(prompt="Hello", strategy_name="strat")