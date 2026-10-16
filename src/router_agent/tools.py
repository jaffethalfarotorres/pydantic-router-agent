"""Deterministic tools, one per actionable intent.

Tools are plain functions from text to ``ToolResult``: no LLM, no network, no
side effects. That is a design choice, not a shortcut: anything a tool reports
(an amount, a date) comes from the input itself, so it can be tested exactly
and cannot be hallucinated. Each tool says what it could not find in
``next_actions`` instead of guessing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from router_agent.envelopes import Intent, ToolResult, ToolStatus


@dataclass(frozen=True)
class Tool:
    name: str
    intent: Intent
    run: Callable[[str], ToolResult]
    description: str


# --- academic metadata -----------------------------------------------------------------

_COURSE_CODE = re.compile(r"\b([A-Z]{2,4}-?\d{3,4})\b")
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_LONG_DATE = re.compile(rf"\b({_MONTHS}) (\d{{1,2}}),? (\d{{4}})\b", re.IGNORECASE)
_POINTS = re.compile(r"\b(\d+)\s*(?:points|pts|marks)\b", re.IGNORECASE)
_BRIEF_CUES = re.compile(
    r"\b(submit|submission|due|deliverable|you must|instructions)\b", re.IGNORECASE
)
_EXAMPLE_CUES = re.compile(
    r"\b(worked example|solution|answer key|sample answer|solved)\b", re.IGNORECASE
)


def _due_date(text: str) -> str | None:
    for match in _ISO_DATE.finditer(text):
        try:
            return date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            continue
    if match := _LONG_DATE.search(text):
        month, day, year = match.groups()
        return datetime.strptime(f"{month.title()} {day} {year}", "%B %d %Y").date().isoformat()
    return None


def _doc_kind(text: str) -> str:
    brief = {cue.lower() for cue in _BRIEF_CUES.findall(text)}
    example = {cue.lower() for cue in _EXAMPLE_CUES.findall(text)}
    if len(brief) > len(example):
        return "assignment_brief"
    if len(example) > len(brief):
        return "worked_example"
    return "unknown"


def extract_academic_metadata(text: str) -> ToolResult:
    """Course code, due date, points and whether it is a brief or a worked example."""
    course = _COURSE_CODE.search(text)
    points = _POINTS.search(text)
    data = {
        "course_code": course.group(1) if course else None,
        "due_date": _due_date(text),
        "points": int(points.group(1)) if points else None,
        "doc_kind": _doc_kind(text),
    }
    next_actions = []
    if data["due_date"] is None:
        next_actions.append("ask the requester for the due date")
    if data["doc_kind"] == "unknown":
        next_actions.append("confirm with a human whether this is a brief or a worked example")

    name = "extract_academic_metadata"
    if data["course_code"] is None and data["due_date"] is None:
        return ToolResult(
            tool=name,
            status=ToolStatus.ERROR,
            summary="no academic metadata found (course code or due date)",
            data=data,
            next_actions=tuple(next_actions),
        )
    summary = f"{data['doc_kind']} for {data['course_code'] or 'unknown course'}"
    summary += f", due {data['due_date'] or 'unknown'}"
    return ToolResult(
        tool=name,
        status=ToolStatus.OK,
        summary=summary,
        data=data,
        next_actions=tuple(next_actions),
    )


# --- invoice total -----------------------------------------------------------------------

_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₡": "CRC"}
_CODES = "USD|EUR|GBP|CRC"
_AMOUNT = re.compile(
    rf"(?P<pre>[{''.join(_SYMBOLS)}]|\b(?:{_CODES})\b)?\s*"
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d{{1,2}})?|\d+(?:\.\d{{1,2}})?)"
    rf"\s*(?P<post>\b(?:{_CODES})\b)?"
)
_TOTAL_KINDS = re.compile(r"\b(grand total|total due|amount due|balance due|total)\b", re.I)
_PRIORITY = {"grand total": 3, "total due": 2, "amount due": 2, "balance due": 2, "total": 1}
_NOT_A_TOTAL = re.compile(r"sub[\s-]?total|total (tax|vat|discount)", re.IGNORECASE)
_SUBTOTAL = re.compile(r"sub[\s-]?total", re.IGNORECASE)
_TAX = re.compile(r"\b(tax|vat|iva)\b", re.IGNORECASE)


def _amount_after(line: str, start: int) -> tuple[Decimal, str | None] | None:
    match = _AMOUNT.search(line, start)
    if not match:
        return None
    marker = match.group("pre") or match.group("post")
    currency = _SYMBOLS.get(marker, marker) if marker else None
    return Decimal(match.group("num").replace(",", "")), currency


def _labelled_amount(lines: list[str], label: re.Pattern[str]) -> Decimal | None:
    for line in lines:
        if (match := label.search(line)) and (found := _amount_after(line, match.end())):
            return found[0]
    return None


def extract_invoice_total(text: str) -> ToolResult:
    """Total, currency, subtotal and tax, plus whether subtotal + tax == total."""
    name = "extract_invoice_total"
    lines = text.splitlines()
    best: tuple[int, int, Decimal, str | None] | None = None
    for index, line in enumerate(lines):
        if _NOT_A_TOTAL.search(line) or not (kind := _TOTAL_KINDS.search(line)):
            continue
        if found := _amount_after(line, kind.end()):
            candidate = (_PRIORITY[kind.group(1).lower()], index, *found)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is None:
        return ToolResult(
            tool=name,
            status=ToolStatus.ERROR,
            summary="no total amount found",
            next_actions=("request a legible copy of the invoice",),
        )

    _, _, total, currency = best
    subtotal = _labelled_amount(lines, _SUBTOTAL)
    tax_lines = [line for line in lines if not _SUBTOTAL.search(line)]
    tax = _labelled_amount(tax_lines, _TAX)
    reconciles = None if subtotal is None or tax is None else subtotal + tax == total

    next_actions = []
    if currency is None:
        next_actions.append("confirm the invoice currency")
    if reconciles is False:
        next_actions.append(f"totals do not reconcile: {subtotal} + {tax} != {total}")
    return ToolResult(
        tool=name,
        status=ToolStatus.OK,
        summary=f"total {total:.2f} {currency or '(currency unknown)'}",
        data={
            "total": f"{total:.2f}",
            "currency": currency,
            "subtotal": f"{subtotal:.2f}" if subtotal is not None else None,
            "tax": f"{tax:.2f}" if tax is not None else None,
            "reconciles": reconciles,
        },
        next_actions=tuple(next_actions),
    )


# --- code snippet ------------------------------------------------------------------------

_FENCE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
_ERROR_TYPE = re.compile(r"^(\w+(?:Error|Exception)):", re.MULTILINE)
_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
_ALIASES = {"py": "python", "js": "javascript", "ts": "typescript"}
_LANGUAGE_HINTS = {
    "python": re.compile(r"\bdef \w+\(|^\s*import \w+|print\(|\bNone\b", re.MULTILINE),
    "javascript": re.compile(r"\bconst \w+|\blet \w+|=>|console\.log|\bfunction \w+\("),
    "sql": re.compile(r"\bSELECT\b.+\bFROM\b", re.IGNORECASE | re.DOTALL),
    "java": re.compile(r"\bpublic (static )?(class|void)\b|System\.out"),
}


def _language(text: str, fence_tag: str | None) -> str | None:
    if fence_tag:
        return _ALIASES.get(fence_tag.lower(), fence_tag.lower())
    for language, hint in _LANGUAGE_HINTS.items():
        if hint.search(text):
            return language
    return None


def extract_code_snippet(text: str) -> ToolResult:
    """Language, fenced snippet size, traceback presence and the raised error type."""
    name = "extract_code_snippet"
    fence = _FENCE.search(text)
    error = _ERROR_TYPE.search(text)
    has_traceback = bool(_TRACEBACK.search(text))
    if not (fence or error or has_traceback):
        return ToolResult(
            tool=name,
            status=ToolStatus.ERROR,
            summary="no code or error output found",
            next_actions=("ask for the code and the full error message",),
        )

    data = {
        "language": _language(text, fence.group(1) if fence else None),
        "has_traceback": has_traceback,
        "error_type": error.group(1) if error else None,
        "snippet_lines": len(fence.group(2).strip().splitlines()) if fence else 0,
    }
    next_actions = []
    if data["error_type"]:
        next_actions.append(f"reproduce the {data['error_type']} with a minimal example")
    if data["language"] is None:
        next_actions.append("ask which language and runtime this is")
    summary = f"{data['language'] or 'unknown language'} snippet"
    summary += f", raises {data['error_type']}" if data["error_type"] else ""
    return ToolResult(
        tool=name,
        status=ToolStatus.OK,
        summary=summary,
        data=data,
        next_actions=tuple(next_actions),
    )


DEFAULT_TOOLS: Mapping[Intent, Tool] = {
    Intent.ACADEMIC_DOC: Tool(
        "extract_academic_metadata",
        Intent.ACADEMIC_DOC,
        extract_academic_metadata,
        "Course code, due date, points and document kind from course material.",
    ),
    Intent.INVOICE: Tool(
        "extract_invoice_total",
        Intent.INVOICE,
        extract_invoice_total,
        "Total, currency, subtotal and tax from an invoice, with a reconciliation check.",
    ),
    Intent.CODE_QUESTION: Tool(
        "extract_code_snippet",
        Intent.CODE_QUESTION,
        extract_code_snippet,
        "Language, snippet and error type from a programming question.",
    ),
}
