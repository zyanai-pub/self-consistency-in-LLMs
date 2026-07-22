import pytest
from unittest.mock import Mock, patch
from src.decoding_strategies.esc import EarlyStoppingSC


@pytest.fixture
def mock_dependencies():
    extractor = Mock()
    model_manager = Mock()
    consensus_manager = Mock()
    return extractor, model_manager, consensus_manager


@pytest.fixture
def esc(mock_dependencies):
    extractor, model_manager, consensus_manager = mock_dependencies
    return EarlyStoppingSC(
        extractor=extractor,
        model_manager=model_manager,
        consensus_manager=consensus_manager
    )


def make_paths(answers: list) -> list:
    return [{"extracted_answer": ans, "confidence": 0.9, "message": f"the answer is {ans}"} for ans in answers]


def test_calculate_entropy_all_same(esc):
    paths = ["11", "11", "11"]
    assert esc._calculate_entropy(paths) == 0.0


def test_calculate_entropy_uniform(esc):
    paths = ["11", "22", "33"]
    entropy = esc._calculate_entropy(paths)
    assert entropy > 0.0


def test_calculate_entropy_empty(esc):
    assert esc._calculate_entropy([]) == 0.0


def test_calculate_entropy_decreases_with_consensus(esc):
    spread = esc._calculate_entropy(["A", "B", "C", "D"])
    majority = esc._calculate_entropy(["A", "A", "A", "B"])
    assert majority < spread


def test_execute_stops_early_when_entropy_low(esc, mock_dependencies):
    _, _, consensus_manager = mock_dependencies

    converged_paths = make_paths(["11"] * 3)
    esc.generate_paths = Mock(return_value=converged_paths)
    consensus_manager.get_majority_vote.return_value = "11"

    result = esc.execute(prompt="prompt", max_paths=15, batch_size=3, entropy_threshold=0.5)

    assert result["paths_sampled"] == 6
    assert result["answer"] == "11"
    assert result["entropy"] == 0.0


def test_execute_runs_to_max_paths_when_no_consensus(esc, mock_dependencies):
    _, _, consensus_manager = mock_dependencies

    diverging_paths = make_paths(["A", "B", "C"])
    esc.generate_paths = Mock(return_value=diverging_paths)
    consensus_manager.get_majority_vote.return_value = "A"

    result = esc.execute(prompt="prompt", max_paths=15, batch_size=3, entropy_threshold=0.1)

    assert result["paths_sampled"] == 15
    assert result["answer"] == "A"


def test_execute_returns_correct_keys(esc, mock_dependencies):
    _, _, consensus_manager = mock_dependencies

    esc.generate_paths = Mock(return_value=make_paths(["11"] * 3))
    consensus_manager.get_majority_vote.return_value = "11"

    result = esc.execute(prompt="prompt", max_paths=15, batch_size=3, entropy_threshold=0.5)

    assert "answer" in result
    assert "paths_sampled" in result
    assert "entropy" in result


def test_execute_entropy_is_none_when_max_paths_below_two_batches(esc, mock_dependencies):
    _, _, consensus_manager = mock_dependencies

    esc.generate_paths = Mock(return_value=make_paths(["11"] * 3))
    consensus_manager.get_majority_vote.return_value = "11"

    result = esc.execute(prompt="prompt", max_paths=3, batch_size=3, entropy_threshold=0.5)

    assert result["entropy"] is None
    assert result["paths_sampled"] == 3


def test_execute_calls_majority_vote_with_all_paths(esc, mock_dependencies):
    _, _, consensus_manager = mock_dependencies

    batch = make_paths(["11", "11", "11"])
    esc.generate_paths = Mock(return_value=batch)
    consensus_manager.get_majority_vote.return_value = "11"

    esc.execute(prompt="prompt", max_paths=15, batch_size=3, entropy_threshold=0.5)

    consensus_manager.get_majority_vote.assert_called_once()
    called_with = consensus_manager.get_majority_vote.call_args[0][0]
    assert all(p in called_with for p in batch)