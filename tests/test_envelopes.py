import json

import pytest
from pydantic import ValidationError

from router_agent.envelopes import (
    SCHEMA_VERSION,
    AgentResult,
    AgentStatus,
    DefenseReport,
    Finding,
    Intent,
    RouteDecision,
    RouteMethod,
    ToolResult,
    ToolStatus,
)

CLEAN = DefenseReport(blocked=False)


def _route(intent: Intent = Intent.INVOICE, confidence: float = 0.9) -> RouteDecision:
    return RouteDecision(
        intent=intent, confidence=confidence, method=RouteMethod.REGEX, rationale="test"
    )


def _tool(status: ToolStatus = ToolStatus.OK) -> ToolResult:
    return ToolResult(tool="extract_invoice_total", status=status, summary="test")


def test_confidence_outside_unit_interval_is_rejected():
    with pytest.raises(ValidationError):
        _route(confidence=1.2)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ToolResult(tool="t", status=ToolStatus.OK, summary="s", surprise=True)


def test_envelopes_are_immutable():
    decision = _route()
    with pytest.raises(ValidationError):
        decision.intent = Intent.OTHER


def test_agent_result_carries_schema_version_in_json():
    result = AgentResult(status=AgentStatus.COMPLETED, defense=CLEAN, route=_route())
    payload = json.loads(result.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert AgentResult.model_validate_json(result.model_dump_json()) == result


def test_blocked_result_cannot_carry_a_route():
    report = DefenseReport(blocked=True, findings=(Finding(rule="r", excerpt="x"),))
    with pytest.raises(ValidationError, match="blocked"):
        AgentResult(status=AgentStatus.BLOCKED, defense=report, route=_route())


def test_blocked_status_requires_a_blocking_defense_report():
    with pytest.raises(ValidationError, match="blocked"):
        AgentResult(status=AgentStatus.BLOCKED, defense=CLEAN)


def test_completed_result_cannot_hide_a_failed_tool():
    with pytest.raises(ValidationError, match="tool_error"):
        AgentResult(
            status=AgentStatus.COMPLETED,
            defense=CLEAN,
            route=_route(),
            tool_result=_tool(ToolStatus.ERROR),
        )


def test_needs_review_result_cannot_report_a_tool_run():
    with pytest.raises(ValidationError, match="needs_review"):
        AgentResult(
            status=AgentStatus.NEEDS_REVIEW,
            defense=CLEAN,
            route=_route(confidence=0.3),
            tool_result=_tool(),
        )


def test_tool_result_requires_a_route():
    with pytest.raises(ValidationError, match="route"):
        AgentResult(status=AgentStatus.COMPLETED, defense=CLEAN, tool_result=_tool())
