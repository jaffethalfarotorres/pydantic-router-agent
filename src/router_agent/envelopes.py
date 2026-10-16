"""Typed envelopes exchanged between the defense layer, the router, the tools and the agent.

Every hand-off in the pipeline is a Pydantic model, never free-form prose, so any
caller (another agent, a UI, a test) can rely on its shape. The models are frozen
and reject unknown fields, and ``AgentResult`` validates its own invariants: a
result cannot claim success while carrying a failed tool, or report a tool run
for an input that was blocked.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

ModelTier = Literal["small", "large"]


class _Envelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Intent(StrEnum):
    """What the input is about. The router may only answer with one of these."""

    ACADEMIC_DOC = "academic_doc"
    INVOICE = "invoice"
    CODE_QUESTION = "code_question"
    OTHER = "other"


class RouteMethod(StrEnum):
    """Which layer produced a routing decision."""

    REGEX = "regex"
    LLM = "llm"
    FALLBACK = "fallback"


class RouteDecision(_Envelope):
    """The router's answer: an intent, how sure it is and how it got there."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    method: RouteMethod
    rationale: str
    model_tier: ModelTier | None = None


class Finding(_Envelope):
    """One prompt-defense rule that matched the input."""

    rule: str
    excerpt: str


class DefenseReport(_Envelope):
    """Outcome of the prompt-defense scan that runs before anything else."""

    blocked: bool
    findings: tuple[Finding, ...] = ()


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class ToolResult(_Envelope):
    """Uniform tool output: status, one-line summary, structured data and next steps."""

    tool: str
    status: ToolStatus
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    next_actions: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"
    TOOL_ERROR = "tool_error"


class AgentResult(_Envelope):
    """Final output of one agent run. Its invariants are checked on construction."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: AgentStatus
    defense: DefenseReport
    route: RouteDecision | None = None
    tool_result: ToolResult | None = None
    trace: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        blocked = self.status is AgentStatus.BLOCKED
        if blocked != self.defense.blocked:
            raise ValueError("status 'blocked' must match the defense report")
        if blocked and (self.route is not None or self.tool_result is not None):
            raise ValueError("a blocked input is never routed nor sent to a tool")
        if self.tool_result is not None and self.route is None:
            raise ValueError("a tool result requires the route that selected the tool")
        if self.status is AgentStatus.NEEDS_REVIEW and self.tool_result is not None:
            raise ValueError("status 'needs_review' means no tool was run")
        tool_failed = self.tool_result is not None and self.tool_result.status is ToolStatus.ERROR
        if tool_failed != (self.status is AgentStatus.TOOL_ERROR):
            raise ValueError("status must be 'tool_error' exactly when the tool failed")
        return self
