"""Intent routing: a deterministic regex layer first, an LLM only when it abstains.

* ``RegexRouter`` answers when the evidence is clear (enough distinct signals
  and a clear margin over the runner-up) and abstains otherwise. It never guesses.
* ``LLMRouter`` asks a model for a JSON verdict and validates it with Pydantic.
  When the output is unusable it diagnoses why, applies a deterministic repair if
  one exists (markdown fences, surrounding prose) and otherwise returns a
  zero-confidence fallback. It never re-calls the model blindly.
* ``HybridRouter`` chains them cheap-first: regex, then the small model tier, and
  one escalation to the large tier only when the small tier answered but was unsure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from router_agent.defense import build_system_prompt, wrap_untrusted
from router_agent.envelopes import Intent, ModelTier, RouteDecision, RouteMethod
from router_agent.providers.base import BackendError, LLMBackend

LARGE_INPUT_CHARS = 4_000
_RATIONALE_CHARS = 200


class Router(Protocol):
    def route(self, text: str) -> RouteDecision: ...


def select_model_tier(text: str, *, override: ModelTier | None = None) -> ModelTier:
    """Cheap-first: the small tier unless the input is long or the caller overrides."""
    if override is not None:
        return override
    return "large" if len(text) > LARGE_INPUT_CHARS else "small"


# --- deterministic layer -------------------------------------------------------------

_I = re.IGNORECASE
_CURRENCY = "$€£₡"  # dollar, euro, pound, colon

_SIGNALS: dict[Intent, dict[str, re.Pattern[str]]] = {
    Intent.INVOICE: {
        "invoice": re.compile(r"\b(invoice|factura)\b", _I),
        "total_due": re.compile(r"\b(total|amount|balance) due\b|\bgrand total\b", _I),
        "subtotal": re.compile(r"\bsubtotal\b", _I),
        "tax": re.compile(r"\b(tax|vat|iva)\b", _I),
        "bill_to": re.compile(r"\bbill(ed)? to\b", _I),
        "payment_terms": re.compile(r"\bpayment terms?\b|\bnet \d{2}\b", _I),
        "currency_amount": re.compile(rf"[{_CURRENCY}]\s?\d"),
    },
    Intent.ACADEMIC_DOC: {
        "coursework": re.compile(r"\b(assignment|homework|coursework|lab report|essay)\b", _I),
        "syllabus": re.compile(r"\b(syllabus|rubric|grading criteria)\b", _I),
        "due_date": re.compile(r"\bdue (date|by|on)\b", _I),
        "course_code": re.compile(r"\b[A-Z]{2,4}-?\d{3,4}\b"),
        "instructor": re.compile(r"\b(professor|instructor|lecturer|teaching assistant)\b", _I),
        "points": re.compile(r"\b\d+\s*(points|pts|marks)\b", _I),
        "submit": re.compile(r"\b(submit|submission)\b", _I),
    },
    Intent.CODE_QUESTION: {
        "code_fence": re.compile(r"```"),
        "traceback": re.compile(r"\btraceback\b|\b\w+(Error|Exception):"),
        "definition": re.compile(
            r"\bdef \w+\(|\bfunction \w+\(|\bclass \w+[(:]|^\s*(import|from) \w+", re.MULTILINE
        ),
        "language": re.compile(
            r"\b(python|javascript|typescript|java|rust|golang|sql)\b|c\+\+", _I
        ),
        "debugging": re.compile(
            r"\b(compile|compiler|debug|refactor|stack trace|null pointer|segfault|bug)\b", _I
        ),
    },
}


@dataclass(frozen=True)
class RegexRouter:
    """Routes on distinct keyword signals; abstains on weak or conflicting evidence."""

    min_signals: int = 2
    min_margin: int = 2

    def route(self, text: str) -> RouteDecision | None:
        hits = {
            intent: [name for name, pattern in signals.items() if pattern.search(text)]
            for intent, signals in _SIGNALS.items()
        }
        ranked = sorted(hits.items(), key=lambda item: len(item[1]), reverse=True)
        (best, best_hits), (_, runner_up_hits) = ranked[0], ranked[1]
        top, second = len(best_hits), len(runner_up_hits)
        if top < self.min_signals:
            return None
        if second and top - second < self.min_margin:
            return None
        return RouteDecision(
            intent=best,
            confidence=round(min(0.95, 0.5 + 0.15 * top), 2),
            method=RouteMethod.REGEX,
            rationale=f"signals: {', '.join(best_hits)}",
        )


# --- LLM layer -----------------------------------------------------------------------


class Diagnosis(StrEnum):
    """Why an LLM reply was (or was not) usable. Recorded in every LLM rationale."""

    OK = "clean_json"
    MARKDOWN_FENCED = "markdown_fenced"
    PROSE_WRAPPED = "prose_wrapped"
    NOT_JSON = "not_json"
    SCHEMA_VIOLATION = "schema_violation"
    UNKNOWN_INTENT = "unknown_intent"
    BACKEND_ERROR = "backend_error"


_INTENT_LIST = ", ".join(intent.value for intent in Intent)
ROUTER_TASK = f"""\
Classify the user content into exactly one intent: {_INTENT_LIST}.
- academic_doc: course material, assignments, syllabi, grading
- invoice: bills, receipts, amounts owed
- code_question: programming questions, code, errors
- other: anything else
Reply with only a JSON object:
{{"intent": "<intent>", "confidence": <number 0 to 1>, "rationale": "<max 20 words>"}}"""


class _Verdict(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


def _extract_json(raw: str) -> tuple[Diagnosis, object]:
    """Find the JSON payload in a reply and say how much repair it needed."""
    stripped = raw.strip()
    try:
        return Diagnosis.OK, json.loads(stripped)
    except json.JSONDecodeError:
        pass
    repaired = Diagnosis.MARKDOWN_FENCED if "```" in stripped else Diagnosis.PROSE_WRAPPED
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"\{", stripped)):
        try:
            payload, _ = decoder.raw_decode(stripped, start)
        except json.JSONDecodeError:
            continue
        return repaired, payload
    return Diagnosis.NOT_JSON, None


class LLMRouter:
    """Asks an LLM for a JSON verdict and validates it; one call per route, always."""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend
        self.system_prompt = build_system_prompt(ROUTER_TASK)

    def route(self, text: str, *, tier: ModelTier = "small") -> RouteDecision:
        try:
            raw = self.backend.complete(wrap_untrusted(text), system=self.system_prompt, tier=tier)
        except BackendError as exc:
            return self._fallback(Diagnosis.BACKEND_ERROR, tier, detail=str(exc))

        diagnosis, payload = _extract_json(raw)
        if diagnosis is Diagnosis.NOT_JSON:
            return self._fallback(diagnosis, tier)
        try:
            verdict = _Verdict.model_validate(payload)
        except ValidationError:
            return self._fallback(Diagnosis.SCHEMA_VIOLATION, tier)
        try:
            intent = Intent(verdict.intent.strip().lower())
        except ValueError:
            return self._fallback(Diagnosis.UNKNOWN_INTENT, tier, detail=verdict.intent)

        return RouteDecision(
            intent=intent,
            confidence=verdict.confidence,
            method=RouteMethod.LLM,
            rationale=f"llm[{diagnosis}]: {verdict.rationale}"[:_RATIONALE_CHARS],
            model_tier=tier,
        )

    @staticmethod
    def _fallback(diagnosis: Diagnosis, tier: ModelTier, *, detail: str = "") -> RouteDecision:
        note = f"llm output unusable [{diagnosis}] {detail}".strip()
        return RouteDecision(
            intent=Intent.OTHER,
            confidence=0.0,
            method=RouteMethod.FALLBACK,
            rationale=f"{note}; not retried"[:_RATIONALE_CHARS],
            model_tier=tier,
        )


# --- hybrid router -------------------------------------------------------------------


class HybridRouter:
    """Regex first, then the small LLM tier, then at most one escalation."""

    def __init__(
        self,
        regex: RegexRouter | None = None,
        llm: LLMRouter | None = None,
        *,
        escalate_below: float = 0.6,
    ) -> None:
        self.regex = regex or RegexRouter()
        self.llm = llm
        self.escalate_below = escalate_below

    def route(self, text: str, *, tier_override: ModelTier | None = None) -> RouteDecision:
        decision = self.regex.route(text)
        if decision is not None:
            return decision
        if self.llm is None:
            return RouteDecision(
                intent=Intent.OTHER,
                confidence=0.0,
                method=RouteMethod.FALLBACK,
                rationale="no deterministic match and no LLM configured",
            )

        tier = select_model_tier(text, override=tier_override)
        decision = self.llm.route(text, tier=tier)
        unsure = decision.method is RouteMethod.LLM and decision.confidence < self.escalate_below
        if tier == "small" and tier_override is None and unsure:
            escalated = self.llm.route(text, tier="large")
            note = f"escalated: small tier confidence {decision.confidence:.2f}"
            return escalated.model_copy(
                update={"rationale": f"{note}; {escalated.rationale}"[:_RATIONALE_CHARS]}
            )
        return decision
