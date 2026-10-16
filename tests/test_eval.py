import json
from itertools import cycle

import pytest

from router_agent.agent import build_agent
from router_agent.envelopes import (
    AgentResult,
    AgentStatus,
    DefenseReport,
    Intent,
    RouteDecision,
    RouteMethod,
)
from router_agent.eval import DEFAULT_CASES_DIR, SUITES, Case, check, load_cases, main, run_suite
from tests import samples


def completed(intent: Intent) -> AgentResult:
    route = RouteDecision(intent=intent, confidence=0.9, method=RouteMethod.LLM, rationale="t")
    return AgentResult(
        status=AgentStatus.COMPLETED, defense=DefenseReport(blocked=False), route=route
    )


def test_check_reports_every_mismatch():
    result = build_agent().run(samples.INVOICE)
    assert check({"status": "completed", "intent": "invoice"}, result) == []
    failures = check({"status": "blocked", "data": {"total": "9.99"}}, result)
    assert len(failures) == 2


def test_check_accepts_a_list_of_allowed_values_and_null_tool():
    result = build_agent().run(samples.NO_SIGNAL)
    assert check({"status": ["needs_review", "completed"], "tool": None}, result) == []


def test_pass_at_k_and_pass_hat_k_separate_flaky_from_reliable():
    flaky = cycle([completed(Intent.INVOICE), completed(Intent.OTHER)])
    case = Case(id="flaky", text="x", expect={"intent": "invoice"})
    report = run_suite("demo", [case], lambda text: next(flaky), k=2)
    assert report.pass_at_k == 1.0
    assert report.pass_hat_k == 0.0


def test_llm_only_cases_are_skipped_without_a_backend():
    case = Case(id="needs-llm", text="x", expect={"intent": "invoice"}, requires_llm=True)
    report = run_suite("demo", [case], build_agent().run, k=1, include_llm=False)
    assert report.skipped == 1
    assert report.ran == 0


@pytest.mark.parametrize("suite", SUITES, ids=lambda suite: suite.name)
def test_shipped_suites_meet_their_release_gates_without_credentials(suite):
    cases = load_cases(DEFAULT_CASES_DIR / suite.file)
    report = run_suite(suite.name, cases, build_agent().run, k=1)
    assert report.ran >= 10
    assert suite.passes(report), report.failures


def test_main_exits_non_zero_when_a_gate_fails(tmp_path, capsys):
    (tmp_path / "router.yaml").write_text(
        "- id: impossible\n  text: hello\n  expect: {status: blocked}\n", encoding="utf-8"
    )
    (tmp_path / "injection.yaml").write_text(
        "- id: ok\n  text: hello\n  expect: {blocked: false}\n", encoding="utf-8"
    )
    assert main(["--cases-dir", str(tmp_path)]) == 1
    assert "impossible" in capsys.readouterr().out


def test_main_writes_a_json_report(tmp_path):
    out = tmp_path / "report.json"
    assert main(["--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert {suite["name"] for suite in report["suites"]} == {"router", "injection"}
    assert report["backend"] == "none"
