import pytest
from unittest.mock import Mock, patch

from src.decoding_strategies.ralu import RaLUSC

@pytest.fixture
def mock_dependencies():
    extractor = Mock()
    model_manager = Mock()
    consensus_manager = Mock()
    return extractor, model_manager, consensus_manager


@pytest.fixture
def ralu(mock_dependencies):
    extractor, model_manager, consensus_manager = mock_dependencies
    return RaLUSC(
        extractor=extractor,
        model_manager=model_manager,
        consensus_manager=consensus_manager,
        max_iters=3,
    )


def make_unit(unit_id: int, code: str = "x = 1", nl: str = "") -> dict:
    return {"unit_id": unit_id, "code": code, "nl_description": nl}


PROMPT = "what is 2 + 2?"

def test_align_unit_returns_immediately_on_ok(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = {"message": "OK the unit is correct.", "confidence": 0.9}

    unit = make_unit(0, "x = 2 + 2")
    result = ralu._align_unit(unit, PROMPT, verified_units=[])

    assert result["code"] == "x = 2 + 2"
    model_manager.generate_inference.assert_called_once()


def test_align_unit_corrects_code_from_fix_block(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    response = "WRONG\n<Fix>x = 4</Fix>\nAnalysis: Addition was wrong."
    model_manager.generate_inference.side_effect = [
        {"message": response, "confidence": 0.7},
        {"message": "OK", "confidence": 0.9},
    ]

    unit = make_unit(0, code="x = 3")
    result = ralu._align_unit(unit, PROMPT, verified_units=[])

    assert result["code"] == "x = 4"
    assert model_manager.generate_inference.call_count == 2


def test_align_unit_extracts_nl_description_from_analysis(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    response = "WRONG\n<Fix>x = 4</Fix>\nAnalysis: Corrected the addition."
    model_manager.generate_inference.side_effect = [
        {"message": response, "confidence": 0.7},
        {"message": "OK", "confidence": 0.9},
    ]

    unit = make_unit(0, code="x = 3")
    result = ralu._align_unit(unit, PROMPT, verified_units=[])

    assert result["nl_description"] == "Corrected the addition."


def test_align_unit_stops_after_max_iters(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = {
        "message": "WRONG\n<Fix>x = 99</Fix>\nAnalysis: still wrong.",
        "confidence": 0.5,
    }

    unit = make_unit(0, code="x = 0")
    result = ralu._align_unit(unit, PROMPT, verified_units=[])

    assert model_manager.generate_inference.call_count == ralu.max_iters
    assert result["code"] == "x = 99"


def test_align_unit_breaks_when_no_fix_block(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = {
        "message": "WRONG but I cannot provide a fix.",
        "confidence": 0.4,
    }

    unit = make_unit(0, code="x = 0")
    result = ralu._align_unit(unit, PROMPT, verified_units=[])

    model_manager.generate_inference.assert_called_once()
    assert result["code"] == "x = 0"


def test_align_unit_returns_original_when_model_returns_none(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = None

    unit = make_unit(0, code="x = 1")
    result = ralu._align_unit(unit, PROMPT, verified_units=[])

    assert result == unit
    model_manager.generate_inference.assert_called_once()


def test_align_unit_preserves_unit_id(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    response = "WRONG\n<Fix>y = 5</Fix>\nAnalysis: Fixed."
    model_manager.generate_inference.side_effect = [
        {"message": response, "confidence": 0.7},
        {"message": "OK", "confidence": 0.9},
    ]

    unit = make_unit(7, code="y = 0")
    result = ralu._align_unit(unit, PROMPT, verified_units=[])

    assert result["unit_id"] == 7

def test_align_unit_includes_verified_units_in_prompt(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = {"message": "OK", "confidence": 0.9}

    verified = [make_unit(0, code="a = 1", nl="first unit")]
    unit = make_unit(1, code="b = 2")
    ralu._align_unit(unit, PROMPT, verified_units=verified)

    call_prompt = model_manager.generate_inference.call_args[0][0]
    assert "a = 1" in call_prompt

def test_synthesise_solution_returns_model_output(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = {"message": "Answer: 4", "confidence": 0.95}

    units = [make_unit(0, "x = 2 + 2", "computes sum")]
    result = ralu._synthesise_solution(units, PROMPT)

    assert result["message"] == "Answer: 4"
    assert result["confidence"] == 0.95


def test_synthesise_solution_returns_fallback_on_none(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = None

    result = ralu._synthesise_solution([], PROMPT)

    assert result["message"] == ""
    assert result["confidence"] == 0.0


def test_synthesise_solution_includes_all_units_in_prompt(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = {"message": "4", "confidence": 0.9}

    units = [
        make_unit(0, "x = 2", "first step"),
        make_unit(1, "y = x + 2", "second step"),
    ]
    ralu._synthesise_solution(units, PROMPT)

    prompt_used = model_manager.generate_inference.call_args[0][0]
    assert "x = 2" in prompt_used
    assert "y = x + 2" in prompt_used
    assert "first step" in prompt_used
    assert "second step" in prompt_used


def test_synthesise_solution_includes_original_prompt(ralu, mock_dependencies):
    _, model_manager, _ = mock_dependencies
    model_manager.generate_inference.return_value = {"message": "4", "confidence": 0.9}

    ralu._synthesise_solution([], PROMPT)

    prompt_used = model_manager.generate_inference.call_args[0][0]
    assert PROMPT in prompt_used

@patch("src.decoding_strategies.ralu.RaLUSC._align_unit")
@patch("src.decoding_strategies.ralu._extract_logic_units")
@patch("src.decoding_strategies.ralu._build_cfg")
def test_generate_ralu_path_returns_correct_keys(mock_build_cfg, mock_extract, mock_align_unit, ralu, mock_dependencies):
    extractor, model_manager, _ = mock_dependencies

    mock_build_cfg.return_value = []
    mock_extract.return_value = [make_unit(0, "x = 4")]
    mock_align_unit.return_value = make_unit(0, "x = 4")
    model_manager.generate_inference.return_value = {"message": "Answer: 4", "confidence": 0.9}
    extractor.extract_from_text.return_value = "4"

    result = ralu._generate_ralu_path(PROMPT)

    assert "extracted_answer" in result
    assert "confidence" in result
    assert "message" in result


@patch("src.decoding_strategies.ralu.RaLUSC._align_unit")
@patch("src.decoding_strategies.ralu._extract_logic_units")
@patch("src.decoding_strategies.ralu._build_cfg")
def test_generate_ralu_path_extracted_answer_from_synthesis(mock_build_cfg, mock_extract, mock_align_unit, ralu, mock_dependencies):
    extractor, model_manager, _ = mock_dependencies

    mock_build_cfg.return_value = []
    mock_extract.return_value = [make_unit(0, "x = 42")]
    mock_align_unit.return_value = make_unit(0, "x = 42")
    model_manager.generate_inference.return_value = {"message": "Answer: 42", "confidence": 0.9}
    extractor.extract_from_text.return_value = "42"

    result = ralu._generate_ralu_path(PROMPT)

    assert result["extracted_answer"] == "42"
    extractor.extract_from_text.assert_called_once_with("Answer: 42")


@patch("src.decoding_strategies.ralu.RaLUSC._generate_ralu_path")
def test_execute_returns_correct_keys(mock_generate_path, ralu, mock_dependencies):
    extractor, model_manager, consensus_manager = mock_dependencies
    mock_generate_path.return_value = {"extracted_answer": "4", "confidence": 0.9, "message": "Answer: 4"}
    model_manager.generate_inference.return_value = {"message": "x = 4\nprint('Answer:', 4)", "confidence": 0.9}
    consensus_manager.get_majority_vote.return_value = "4"

    result = ralu.execute(PROMPT, num_paths=2)

    assert "answer" in result
    assert "paths_sampled" in result


@patch("src.decoding_strategies.ralu.RaLUSC._generate_ralu_path")
def test_execute_paths_sampled_matches_num_paths(mock_generate_path, ralu, mock_dependencies):
    extractor, model_manager, consensus_manager = mock_dependencies
    mock_generate_path.return_value = {"extracted_answer": "4", "confidence": 0.9}
    consensus_manager.get_majority_vote.return_value = "4"

    result = ralu.execute(PROMPT, num_paths=3)

    assert result["paths_sampled"] == 3


@patch("src.decoding_strategies.ralu.RaLUSC._generate_ralu_path")
def test_execute_calls_majority_vote_with_all_paths(mock_generate_path, ralu, mock_dependencies):
    extractor, model_manager, consensus_manager = mock_dependencies
    mock_generate_path.return_value = {"extracted_answer": "4", "confidence": 0.9}
    consensus_manager.get_majority_vote.return_value = "4"

    ralu.execute(PROMPT, num_paths=3)

    args = consensus_manager.get_majority_vote.call_args[0][0]
    assert len(args) == 3


@patch("src.decoding_strategies.ralu.RaLUSC._generate_ralu_path")
def test_execute_answer_comes_from_majority_vote(mock_generate_path, ralu, mock_dependencies):
    extractor, model_manager, consensus_manager = mock_dependencies
    mock_generate_path.return_value = {"extracted_answer": "4", "confidence": 0.9}
    consensus_manager.get_majority_vote.return_value = "the_answer"

    result = ralu.execute(PROMPT, num_paths=2)

    assert result["answer"] == "the_answer"