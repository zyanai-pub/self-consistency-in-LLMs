import pytest
from unittest.mock import Mock
from itertools import cycle

from src.evaluation_module.extractor import AnswerExtractor
from src.evaluation_module.consensus import ConsensusManager
from src.decoding_strategies.baseline_sc import BaselineSC
from src.decoding_strategies.esc import EarlyStoppingSC
from src.decoding_strategies.seer_sc import SeerSC
from src.controller.framework_controller import FrameworkController


def paths_from_mock_data(mock_data: dict) -> list:
    return [
        {
            "extracted_answer": path.get("extracted_answer"),
            "confidence":       path.get("confidence_score", 0.0),
        }
        for path in mock_data["generated_paths"]
    ]


def model_manager_from_mock(mock_data: dict) -> Mock:
    model_manager = Mock()
    responses = iter([
        {
            "message":    path.get("raw_text", ""),
            "confidence": path.get("confidence_score", 0.5),
        }
        for path in mock_data["generated_paths"]
        if path.get("raw_text")
    ])
    model_manager.generate_inference.side_effect = lambda *a, **kw: next(responses, None)
    return model_manager

# Baseline end-to-end tests
class TestBaselineSCIntegration:
    def test_basic_consensus_produces_correct_answer(self, load_mock_data):
        data = load_mock_data("basic_consensus.json")
        model_manager = model_manager_from_mock(data)
        extractor = AnswerExtractor()
        consensus = ConsensusManager()
        strategy = BaselineSC(extractor, model_manager, consensus)

        result = strategy.execute(data["prompt"], num_samples=len(data["generated_paths"]))

        assert result["answer"] == data["expected_truth"]

    def test_low_entropy_correct_answer_unanimous(self, load_mock_data):
        data = load_mock_data("low_entropy.json")
        model_manager = model_manager_from_mock(data)
        extractor = AnswerExtractor()
        consensus = ConsensusManager()
        strategy = BaselineSC(extractor, model_manager, consensus)

        result = strategy.execute(data["prompt"], num_samples=len(data["generated_paths"]))

        assert result["answer"] == data["expected_truth"]
        assert result["paths_sampled"] == len(data["generated_paths"])

    def test_malformed_output_still_returns_valid_answer(self, load_mock_data):
        data = load_mock_data("malformed_output.json")
        model_manager = model_manager_from_mock(data)
        extractor = AnswerExtractor()
        consensus = ConsensusManager()
        strategy = BaselineSC(extractor, model_manager, consensus)

        result = strategy.execute(data["prompt"], num_samples=2)

        assert result["answer"] == data["expected_truth"]

    def test_generate_inference_called_once_per_sample(self, load_mock_data):
        data = load_mock_data("low_entropy.json")
        model_manager = model_manager_from_mock(data)
        extractor = AnswerExtractor()
        consensus = ConsensusManager()
        strategy = BaselineSC(extractor, model_manager, consensus)

        strategy.execute(data["prompt"], num_samples=4)

        assert model_manager.generate_inference.call_count == 4

# Test end-to-end esc
class TestEarlyStoppingSCIntegration:
    def test_low_entropy_stops_before_max_paths(self, load_mock_data):
        data = load_mock_data("low_entropy.json")
        responses = cycle([
            {"message": p["raw_text"], "confidence": p["confidence_score"]}
            for p in data["generated_paths"]
        ])
        model_manager = Mock()
        model_manager.generate_inference.side_effect = lambda *a, **kw: next(responses)
        extractor = AnswerExtractor()
        consensus = ConsensusManager()
        strategy = EarlyStoppingSC(extractor, model_manager, consensus)

        result = strategy.execute(
            data["prompt"],
            max_paths=15,
            batch_size=3,
            entropy_threshold=0.5,
        )

        assert result["answer"] == data["expected_truth"]
        assert result["paths_sampled"] < 15
        assert result["entropy"] == 0.0

    def test_high_entropy_runs_to_max_paths(self, load_mock_data):
        data = load_mock_data("high_entropy_esc.json")
        responses = [
            {"message": p["raw_text"], "confidence": p["confidence_score"]}
            for p in data["generated_paths"]
        ] * 5
        model_manager = Mock()
        model_manager.generate_inference.side_effect = iter(responses)
        extractor = AnswerExtractor()
        consensus = ConsensusManager()
        strategy = EarlyStoppingSC(extractor, model_manager, consensus)

        result = strategy.execute(
            data["prompt"],
            max_paths=12,
            batch_size=3,
            entropy_threshold=0.01
        )

        assert result["paths_sampled"] == 12

    def test_entropy_key_present_and_is_float(self, load_mock_data):
        data = load_mock_data("low_entropy.json")
        model_manager = Mock()
        model_manager.generate_inference.return_value = {
            "message":    data["generated_paths"][0]["raw_text"],
            "confidence": 0.9,
        }
        extractor = AnswerExtractor()
        consensus = ConsensusManager()
        strategy = EarlyStoppingSC(extractor, model_manager, consensus)

        result = strategy.execute(data["prompt"], max_paths=6, batch_size=3, entropy_threshold=0.5)

        assert "entropy" in result
        assert isinstance(result["entropy"], float)

# Test seer end-to-end
class TestSeerSCIntegration:

    @staticmethod
    def _make_seer(sys1_data, sys2_data):
        return SeerSC(
            extractor=AnswerExtractor(),
            system1_model_manager=model_manager_from_mock(sys1_data),
            system2_model_manager=model_manager_from_mock(sys2_data),
            consensus_manager=ConsensusManager(),
        )

    def test_low_entropy_system1_allocates_minimal_budget(self, load_mock_data):
        low = load_mock_data("low_entropy.json")
        basic = load_mock_data("basic_consensus.json")
        seer = self._make_seer(low, basic)
        result = seer.execute(low["prompt"], m=4, n=15)
        assert result["paths_sampled"] == 1

    def test_low_entropy_produces_correct_final_answer(self, load_mock_data):
        low = load_mock_data("low_entropy.json")
        low_s2 = load_mock_data("system2_mock_data_low_ent.json")
        seer = self._make_seer(low, low_s2)
        result = seer.execute(low["prompt"], m=4, n=15)
        assert result["answer"] == low["expected_truth"]

    def test_system1_entropy_value_present_in_result(self, load_mock_data):
        low = load_mock_data("low_entropy.json")
        basic = load_mock_data("basic_consensus.json")
        seer = self._make_seer(low, basic)
        result = seer.execute(low["prompt"], m=4, n=10)
        assert "system1_entropy" in result
        assert result["system1_entropy"] >= 0.0

class TestFrameworkControllerIntegration:

    @pytest.fixture
    def controller(self, load_mock_data):
        data = load_mock_data("low_entropy.json")
        model_manager = Mock()
        model_manager.generate_inference.return_value = {
            "message":    data["generated_paths"][0]["raw_text"],
            "confidence": 0.9,
        }
        sys1_model_manager = Mock()
        sys1_model_manager.generate_inference.return_value = {
            "message":    data["generated_paths"][0]["raw_text"],
            "confidence": 0.9,
        }
        return FrameworkController(
            model_manager=model_manager,
            extractor=AnswerExtractor(),
            consensus_builder=ConsensusManager(),
            system1_model_manager=sys1_model_manager,
        ), data["prompt"]

    @pytest.mark.parametrize("strategy,kwargs", [
        ("baseline", {"num_samples": 4}),
        ("esc",      {"max_paths": 6, "batch_size": 3, "entropy_threshold": 0.5}),
        ("seer",     {"m": 4, "n": 6}),
    ])
    def test_strategy_returns_answer_and_timing(self, controller, strategy, kwargs):
        ctrl, prompt = controller
        result = ctrl.execute_task(prompt, strategy_name=strategy, **kwargs)

        assert "answer" in result
        assert "time_seconds" in result
        assert result["time_seconds"] >= 0
        assert result["strategy"] == strategy

    def test_unknown_strategy_raises_value_error(self, controller):
        ctrl, prompt = controller
        with pytest.raises(ValueError, match="Unknown strategy"):
            ctrl.execute_task(prompt, strategy_name="nonexistent")

    def test_strategy_name_is_case_insensitive(self, controller):
        ctrl, prompt = controller
        result = ctrl.execute_task(prompt, strategy_name="BASELINE", num_samples=3)
        assert "answer" in result

    def test_split_vote_correct_answer_via_controller(self, load_mock_data):
        data = load_mock_data("split_vote_consensus.json")
        model_manager = model_manager_from_mock(data)
        ctrl = FrameworkController(
            model_manager=model_manager,
            extractor=AnswerExtractor(),
            consensus_builder=ConsensusManager(),
            system1_model_manager=Mock(),
        )

        result = ctrl.execute_task(
            data["prompt"],
            strategy_name="baseline",
            num_samples=len(data["generated_paths"]),
        )

        assert result["answer"] == data["expected_truth"]