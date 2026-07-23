import math
import pytest
from unittest.mock import Mock
from src.decoding_strategies.seer_sc import SeerSC


@pytest.fixture
def mock_dependencies():
    extractor = Mock()
    system1_model_manager = Mock()
    system2_model_manager = Mock()
    consensus_manager = Mock()
    return extractor, system1_model_manager, system2_model_manager, consensus_manager


@pytest.fixture
def seer(mock_dependencies):
    extractor, system1_mm, system2_mm, consensus_manager = mock_dependencies
    return SeerSC(
        extractor=extractor,
        system1_model_manager=system1_mm,
        system2_model_manager=system2_mm,
        consensus_manager=consensus_manager
    )


def make_paths(answers: list, confidence: float = 0.9) -> list:
    return [{"extracted_answer": ans, "confidence": confidence, "entropy": 0.1, "message": f"the answer is {ans}"} for ans in answers]


def test_entropy_all_same_answer(seer):
    paths = make_paths(["42", "42", "42"])
    assert seer._compute_confidence_weighted_entropy(paths) == 0.0


def test_entropy_uniform_distribution(seer):
    paths = make_paths(["A", "B", "C"])
    entropy = seer._compute_confidence_weighted_entropy(paths)
    assert abs(entropy - math.log(3)) < 1e-9


def test_entropy_empty_paths(seer):
    assert seer._compute_confidence_weighted_entropy([]) == 0.0


def test_entropy_all_none_answers(seer):
    paths = [{"extracted_answer": None, "confidence": 0.9, "entropy": 0.1, "message": "?"}] * 3
    assert seer._compute_confidence_weighted_entropy(paths) == 0.0


def test_entropy_decreases_with_consensus(seer):
    spread = seer._compute_confidence_weighted_entropy(make_paths(["A", "B", "C", "D"]))
    majority = seer._compute_confidence_weighted_entropy(make_paths(["A", "A", "A", "B"]))
    assert majority < spread


def test_entropy_weighted_by_confidence(seer):
    paths = [
        {"extracted_answer": "A", "confidence": 0.99, "entropy": 0.1, "message": "A"},
        {"extracted_answer": "B", "confidence": 0.01, "entropy": 0.1, "message": "B"},
    ]
    entropy = seer._compute_confidence_weighted_entropy(paths)
    uniform_entropy = math.log(2)
    assert entropy < uniform_entropy


def test_allocate_budget_low_entropy_returns_1(seer):
    m, n = 5, 15
    low_entropy = (1 / 10) * math.log(m) * 0.5  # well below tau1
    assert seer._allocate_budget(low_entropy, m, n) == 1


def test_allocate_budget_mid_entropy_returns_half(seer):
    m, n = 5, 15
    tau1 = (1 / 10) * math.log(m)
    tau2 = (1 / 3) * math.log(m)
    mid_entropy = (tau1 + tau2) / 2
    assert seer._allocate_budget(mid_entropy, m, n) == n // 2


def test_allocate_budget_high_entropy_returns_n(seer):
    m, n = 5, 15
    high_entropy = math.log(m)  # maximum possible entropy
    assert seer._allocate_budget(high_entropy, m, n) == n


def test_execute_returns_correct_keys(seer, mock_dependencies):
    extractor, system1_mm, _, consensus_manager = mock_dependencies

    system1_mm.generate_inference.return_value = {"message": "msg", "confidence": 0.9, "entropy": 0.1}
    extractor.extract_from_text.return_value = "42"
    seer.generate_paths = Mock(return_value=make_paths(["42"] * 3))
    consensus_manager.get_majority_vote.return_value = "42"

    result = seer.execute(prompt="What is 6x7?", m=3, n=9)

    assert "answer" in result
    assert "paths_sampled" in result
    assert "system1_entropy" in result


def test_execute_low_entropy_allocates_budget_1(seer, mock_dependencies):
    extractor, system1_mm, _, consensus_manager = mock_dependencies

    system1_mm.generate_inference.return_value = {"message": "msg", "confidence": 0.9, "entropy": 0.1}
    extractor.extract_from_text.return_value = "42"
    seer.generate_paths = Mock(return_value=make_paths(["42"]))
    consensus_manager.get_majority_vote.return_value = "42"

    result = seer.execute(prompt="What is 6x7?", m=5, n=15)

    seer.generate_paths.assert_called_once_with("What is 6x7?", num_samples=1)
    assert result["paths_sampled"] == 1


def test_execute_high_entropy_allocates_full_budget(seer, mock_dependencies):
    extractor, system1_mm, _, consensus_manager = mock_dependencies

    system1_mm.generate_inference.return_value = {"message": "msg", "confidence": 0.9, "entropy": 0.5}
    answers = ["A", "B", "C", "D", "E"]
    extractor.extract_from_text.side_effect = answers
    seer.generate_paths = Mock(return_value=make_paths(answers * 3))
    consensus_manager.get_majority_vote.return_value = "A"

    result = seer.execute(prompt="hard problem", m=5, n=15)

    seer.generate_paths.assert_called_once_with("hard problem", num_samples=15)
    assert result["paths_sampled"] == 15


def test_execute_none_system1_responses_handled(seer, mock_dependencies):
    extractor, system1_mm, _, consensus_manager = mock_dependencies

    system1_mm.generate_inference.side_effect = [None, None,
        {"message": "msg", "confidence": 0.9, "entropy": 0.1},
        {"message": "msg", "confidence": 0.9, "entropy": 0.1},
        {"message": "msg", "confidence": 0.9, "entropy": 0.1},
    ]
    extractor.extract_from_text.return_value = "42"
    seer.generate_paths = Mock(return_value=make_paths(["42"]))
    consensus_manager.get_majority_vote.return_value = "42"

    result = seer.execute(prompt="prompt", m=5, n=15)
    assert result["answer"] == "42"


def test_execute_calls_majority_vote_with_system2_paths(seer, mock_dependencies):
    extractor, system1_mm, _, consensus_manager = mock_dependencies

    system1_mm.generate_inference.return_value = {"message": "msg", "confidence": 0.9, "entropy": 0.1}
    extractor.extract_from_text.return_value = "42"
    system2_paths = make_paths(["42", "42", "7"])
    seer.generate_paths = Mock(return_value=system2_paths)
    consensus_manager.get_majority_vote.return_value = "42"

    seer.execute(prompt="prompt", m=5, n=15)

    consensus_manager.get_majority_vote.assert_called_once_with(system2_paths)