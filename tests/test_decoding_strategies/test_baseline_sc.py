import json
from pathlib import Path

import pytest
from unittest.mock import Mock
from src.decoding_strategies.baseline_sc import BaselineSC

@pytest.fixture
def mock_dependencies():
    extractor = Mock()
    model_manager = Mock()
    consensus_manager = Mock()
    return extractor, model_manager, consensus_manager


@pytest.fixture
def baseline_sc(mock_dependencies):
    extractor, model_manager, consensus_manager = mock_dependencies
    return BaselineSC(
        extractor=extractor,
        model_manager=model_manager,
        consensus_manager=consensus_manager
    )

def test_generate_paths_structure_and_calls(baseline_sc, mock_dependencies):
    extractor, model_manager, _ = mock_dependencies

    model_manager.generate_inference.return_value = {
        'message': "the answer is 6",
        'confidence': 0.99
    }
    extractor.extract_from_text.return_value = "6"

    prompt = "What is 6 times 1?"
    num_samples = 3

    paths = baseline_sc.generate_paths(prompt, num_samples=3, temperature=0.7)

    assert len(paths) == num_samples
    assert model_manager.generate_inference.call_count == num_samples
    assert extractor.extract_from_text.call_count == num_samples

    for path in paths:
        assert path['extracted_answer'] == "6"
        assert path['confidence'] == 0.99
        assert path['message'] == "the answer is 6"


def test_apply_majority_vote_with_mock_data(baseline_sc, mock_dependencies):
    current_file_path = Path(__file__)
    mock_file_path = current_file_path.parent.parent / 'mock_data' / 'basic_consensus.json'
    _, _, consensus_manager = mock_dependencies

    with open(mock_file_path, 'r') as f:
        mock_paths = json.load(f)

    expected_answer = "11"
    consensus_manager.get_majority_vote.return_value = expected_answer

    result = baseline_sc._apply_majority_vote(mock_paths)

    consensus_manager.get_majority_vote.assert_called_once_with(mock_paths)
    assert result == expected_answer


def test_generate_paths_handles_missing_confidence(baseline_sc, mock_dependencies):
    extractor, model_manager, _ = mock_dependencies

    model_manager.generate_inference.return_value = {
        'message': "the answer is 10"
        # Missing confidence
    }
    extractor.extract_from_text.return_value = "10"

    paths = baseline_sc.generate_paths("prompt", num_samples=1)

    assert paths[0]['extracted_answer'] == "10"
    assert paths[0]['confidence'] is None