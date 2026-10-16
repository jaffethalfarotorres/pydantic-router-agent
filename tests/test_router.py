import json

import pytest

from router_agent.defense import DEFENSE_HEADER
from router_agent.envelopes import Intent, RouteMethod
from router_agent.providers import MockBackend
from router_agent.router import (
    LARGE_INPUT_CHARS,
    Diagnosis,
    HybridRouter,
    LLMRouter,
    RegexRouter,
    select_model_tier,
)
from tests import samples
from tests.fakes import all_backends


def verdict(intent: str = "invoice", confidence: float = 0.9) -> str:
    return json.dumps({"intent": intent, "confidence": confidence, "rationale": "test"})


# --- deterministic layer -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        (samples.INVOICE, Intent.INVOICE),
        (samples.ACADEMIC, Intent.ACADEMIC_DOC),
        (samples.CODE, Intent.CODE_QUESTION),
    ],
)
def test_regex_router_resolves_clear_inputs(text, intent):
    decision = RegexRouter().route(text)
    assert decision is not None
    assert decision.intent is intent
    assert decision.method is RouteMethod.REGEX
    assert decision.confidence >= 0.8


@pytest.mark.parametrize("text", [samples.NO_SIGNAL, samples.WEAK_SIGNAL, samples.CONFLICT])
def test_regex_router_abstains_instead_of_guessing(text):
    assert RegexRouter().route(text) is None


def test_regex_rationale_names_the_matched_signals():
    decision = RegexRouter().route(samples.INVOICE)
    assert "subtotal" in decision.rationale


# --- LLM layer ---------------------------------------------------------------------


def test_llm_router_sends_defended_prompt_with_fenced_input():
    backend = MockBackend([verdict()])
    LLMRouter(backend).route("pay me")
    call = backend.calls[0]
    assert call.system.startswith(DEFENSE_HEADER)
    assert "<untrusted_input>\npay me\n</untrusted_input>" in call.prompt


def test_llm_router_accepts_clean_json():
    decision = LLMRouter(MockBackend([verdict("code_question", 0.7)])).route("x")
    assert decision.intent is Intent.CODE_QUESTION
    assert decision.confidence == 0.7
    assert decision.method is RouteMethod.LLM
    assert Diagnosis.OK in decision.rationale


@pytest.mark.parametrize(
    ("raw", "diagnosis"),
    [
        (f"```json\n{verdict()}\n```", Diagnosis.MARKDOWN_FENCED),
        (f"Sure! Here is the classification: {verdict()} Hope it helps.", Diagnosis.PROSE_WRAPPED),
    ],
)
def test_llm_router_recovers_common_format_slips_without_a_second_call(raw, diagnosis):
    backend = MockBackend([raw])
    decision = LLMRouter(backend).route("x")
    assert decision.intent is Intent.INVOICE
    assert decision.method is RouteMethod.LLM
    assert diagnosis in decision.rationale
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    ("raw", "diagnosis"),
    [
        ("I think this is an invoice.", Diagnosis.NOT_JSON),
        ('{"intent": "invoice"', Diagnosis.NOT_JSON),
        (json.dumps({"intent": "invoice", "confidence": 3}), Diagnosis.SCHEMA_VIOLATION),
        (json.dumps(["invoice", 0.9]), Diagnosis.SCHEMA_VIOLATION),
        (verdict("purchase_order"), Diagnosis.UNKNOWN_INTENT),
    ],
)
def test_unusable_llm_output_falls_back_without_blind_retry(raw, diagnosis):
    backend = MockBackend([raw])
    decision = LLMRouter(backend).route("x")
    assert decision.intent is Intent.OTHER
    assert decision.confidence == 0.0
    assert decision.method is RouteMethod.FALLBACK
    assert diagnosis in decision.rationale
    assert len(backend.calls) == 1


def test_provider_outage_falls_back_instead_of_raising():
    backend = MockBackend(fail_with="503 service unavailable")
    decision = LLMRouter(backend).route("x")
    assert decision.method is RouteMethod.FALLBACK
    assert Diagnosis.BACKEND_ERROR in decision.rationale
    assert "503" in decision.rationale


def test_every_provider_yields_the_identical_decision():
    raw = f"```json\n{verdict('academic_doc', 0.85)}\n```"
    decisions = {
        name: LLMRouter(backend).route("a syllabus") for name, backend in all_backends(raw).items()
    }
    assert len(set(decisions.values())) == 1, decisions


# --- hybrid router and cost tiers -----------------------------------------------------


def test_hybrid_router_never_calls_the_llm_when_regex_is_confident():
    backend = MockBackend([])
    decision = HybridRouter(llm=LLMRouter(backend)).route(samples.INVOICE)
    assert decision.method is RouteMethod.REGEX
    assert backend.calls == []


def test_hybrid_router_falls_back_to_the_small_llm_tier():
    backend = MockBackend([verdict("invoice", 0.9)])
    decision = HybridRouter(llm=LLMRouter(backend)).route(samples.WEAK_SIGNAL)
    assert decision.method is RouteMethod.LLM
    assert decision.model_tier == "small"
    assert [call.model for call in backend.calls] == ["mock-small"]


def test_hybrid_router_escalates_once_when_the_small_tier_is_unsure():
    backend = MockBackend([verdict("invoice", 0.4), verdict("invoice", 0.9)])
    decision = HybridRouter(llm=LLMRouter(backend), escalate_below=0.6).route(samples.NO_SIGNAL)
    assert [call.model for call in backend.calls] == ["mock-small", "mock-large"]
    assert decision.confidence == 0.9
    assert decision.model_tier == "large"
    assert "escalated" in decision.rationale


def test_hybrid_router_does_not_escalate_malformed_output():
    backend = MockBackend(["not json at all"])
    decision = HybridRouter(llm=LLMRouter(backend)).route(samples.NO_SIGNAL)
    assert decision.method is RouteMethod.FALLBACK
    assert len(backend.calls) == 1


def test_hybrid_router_without_llm_reports_a_zero_confidence_fallback():
    decision = HybridRouter().route(samples.NO_SIGNAL)
    assert decision.method is RouteMethod.FALLBACK
    assert decision.confidence == 0.0


def test_tier_override_is_respected_and_disables_escalation():
    backend = MockBackend([verdict("invoice", 0.3)])
    router = HybridRouter(llm=LLMRouter(backend))
    decision = router.route(samples.NO_SIGNAL, tier_override="large")
    assert [call.model for call in backend.calls] == ["mock-large"]
    assert decision.confidence == 0.3


def test_model_tier_is_cheap_first():
    assert select_model_tier("short input") == "small"
    assert select_model_tier("x" * (LARGE_INPUT_CHARS + 1)) == "large"
    assert select_model_tier("short input", override="large") == "large"
