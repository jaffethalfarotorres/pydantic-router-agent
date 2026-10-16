"""The agent: defense -> route -> code gate -> tool -> typed result.

The LLM decides only *what the input is* (an ``Intent`` from a closed enum).
Code decides *whether a tool runs and which one*:

* the input must pass the prompt-defense scan before anything else sees it;
* the route must clear a confidence threshold, otherwise a human reviews it;
* the tool comes from a fixed registry keyed by intent, never from model text;
* the tool must return a ``ToolResult`` that names itself, or the run fails loudly.

Every run returns an ``AgentResult`` whose own validators reject contradictory
states, plus a short trace of each step that fired.
"""

from __future__ import annotations

from collections.abc import Mapping

from router_agent.defense import DEFAULT_MAX_CHARS, scan
from router_agent.envelopes import (
    AgentResult,
    AgentStatus,
    Intent,
    ToolResult,
    ToolStatus,
)
from router_agent.providers.base import LLMBackend
from router_agent.router import HybridRouter, LLMRouter, Router
from router_agent.tools import DEFAULT_TOOLS, Tool


class ToolRegistryError(ValueError):
    """The tool registry maps an intent to a tool built for another intent."""


class ToolContractError(RuntimeError):
    """A tool returned something other than its own ``ToolResult``."""


class Agent:
    def __init__(
        self,
        router: Router,
        tools: Mapping[Intent, Tool] = DEFAULT_TOOLS,
        *,
        min_confidence: float = 0.6,
        max_input_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        for intent, tool in tools.items():
            if tool.intent is not intent:
                raise ToolRegistryError(
                    f"{tool.name} handles {tool.intent.value}, registered under {intent.value}"
                )
        self.router = router
        self.tools = dict(tools)
        self.min_confidence = min_confidence
        self.max_input_chars = max_input_chars

    def run(self, text: str) -> AgentResult:
        report = scan(text, max_chars=self.max_input_chars)
        if report.blocked:
            rules = ", ".join(finding.rule for finding in report.findings)
            return AgentResult(
                status=AgentStatus.BLOCKED, defense=report, trace=(f"defense: blocked ({rules})",)
            )
        trace = ["defense: clean"]

        decision = self.router.route(text)
        trace.append(
            f"route: {decision.intent.value} via {decision.method.value} "
            f"({decision.confidence:.2f})"
        )
        if decision.confidence < self.min_confidence:
            trace.append(f"gate: hold ({decision.confidence:.2f} < {self.min_confidence:.2f})")
            return AgentResult(
                status=AgentStatus.NEEDS_REVIEW, defense=report, route=decision, trace=tuple(trace)
            )
        trace.append(f"gate: pass ({decision.confidence:.2f} >= {self.min_confidence:.2f})")

        tool = self.tools.get(decision.intent)
        if tool is None:
            trace.append(f"tool: no tool registered for {decision.intent.value}")
            return AgentResult(
                status=AgentStatus.COMPLETED, defense=report, route=decision, trace=tuple(trace)
            )

        result = self._execute(tool, text)
        trace.append(f"tool: {tool.name} -> {result.status.value}")
        status = AgentStatus.COMPLETED if result.status is ToolStatus.OK else AgentStatus.TOOL_ERROR
        return AgentResult(
            status=status,
            defense=report,
            route=decision,
            tool_result=result,
            trace=tuple(trace),
        )

    @staticmethod
    def _execute(tool: Tool, text: str) -> ToolResult:
        try:
            result = tool.run(text)
        except Exception as exc:
            return ToolResult(
                tool=tool.name,
                status=ToolStatus.ERROR,
                summary=f"tool raised {type(exc).__name__}: {exc}",
            )
        if not isinstance(result, ToolResult):
            raise ToolContractError(
                f"{tool.name} returned {type(result).__name__}, expected ToolResult"
            )
        if result.tool != tool.name:
            raise ToolContractError(f"{tool.name} returned a result labelled {result.tool}")
        return result


def build_agent(
    backend: LLMBackend | None = None,
    *,
    min_confidence: float = 0.6,
    max_input_chars: int = DEFAULT_MAX_CHARS,
) -> Agent:
    """Regex-only agent by default; pass a backend to enable the LLM fallback."""
    llm = LLMRouter(backend) if backend is not None else None
    return Agent(
        HybridRouter(llm=llm), min_confidence=min_confidence, max_input_chars=max_input_chars
    )
