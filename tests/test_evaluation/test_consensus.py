import pytest
from evaluation_module.consensus import ConsensusManager

@pytest.fixture
def consensus_manager():
    return ConsensusManager()

def test_get_majority_vote_empty_generated_paths(consensus_manager):
    assert consensus_manager.get_majority_vote({}) == ""

def test_get_majority_vote_no_valid_extracted_answers(consensus_manager):
    paths = [
        {"irrelevant_key": "answer"},
        {"extracted_answer": None},
        {"extracted_answer": None}
    ]
    assert consensus_manager.get_majority_vote(paths) == ""

def test_get_majority_vote_clear_majority(consensus_manager):
    paths = [
        {"extracted_answer": "A"},
        {"extracted_answer": "B"},
        {"extracted_answer": "A"}
    ]
    assert consensus_manager.get_majority_vote(paths) == "A"

def test_get_majority_vote_tie_broken_by_confidence_score(consensus_manager):
    paths = [
        {"extracted_answer": "A", "confidence_score": 0.8},
        {"extracted_answer": "A", "confidence_score": 0.6}, # Average A: 0.7
        {"extracted_answer": "B", "confidence_score": 0.9},
        {"extracted_answer": "B", "confidence_score": 0.9}  # Average B: 0.9
    ]
    assert consensus_manager.get_majority_vote(paths) == "B"

def test_get_majority_vote_tie_with_negative_confidence_scores(consensus_manager):
    paths = [
        {"extracted_answer": "A", "confidence_score": -1.5},
        {"extracted_answer": "A", "confidence_score": -0.5},
        {"extracted_answer": "B", "confidence_score": -2.0},
        {"extracted_answer": "B", "confidence_score": -3.0}
    ]
    assert consensus_manager.get_majority_vote(paths) == "A"