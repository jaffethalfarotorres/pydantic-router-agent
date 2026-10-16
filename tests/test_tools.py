import pytest

from router_agent.envelopes import Intent, ToolStatus
from router_agent.tools import (
    DEFAULT_TOOLS,
    extract_academic_metadata,
    extract_code_snippet,
    extract_invoice_total,
)
from tests import samples

# --- academic metadata -----------------------------------------------------------------


def test_academic_brief_metadata_is_extracted():
    result = extract_academic_metadata(samples.ACADEMIC)
    assert result.status is ToolStatus.OK
    assert result.data["course_code"] == "CS-101"
    assert result.data["due_date"] == "2026-11-20"
    assert result.data["points"] == 20
    assert result.data["doc_kind"] == "assignment_brief"


def test_worked_example_is_told_apart_from_an_assignment_brief():
    text = "CS-101 Worked example: solution to Exercise 2. The answer key follows."
    assert extract_academic_metadata(text).data["doc_kind"] == "worked_example"


def test_ambiguous_document_kind_is_escalated_to_a_human():
    result = extract_academic_metadata("MAT-210 Submit the solution by 2026-11-20.")
    assert result.data["doc_kind"] == "unknown"
    assert any("human" in action for action in result.next_actions)


def test_long_form_due_date_is_normalized_to_iso():
    result = extract_academic_metadata("ENG-205 essay, due by November 3, 2026.")
    assert result.data["due_date"] == "2026-11-03"


def test_impossible_date_is_not_reported():
    result = extract_academic_metadata("CS-101 homework due on 2026-13-45")
    assert result.data["due_date"] is None
    assert any("due date" in action for action in result.next_actions)


def test_text_without_academic_metadata_is_a_tool_error():
    result = extract_academic_metadata("Lovely weather today.")
    assert result.status is ToolStatus.ERROR


# --- invoice total -----------------------------------------------------------------------


def test_invoice_total_currency_and_breakdown_are_extracted():
    result = extract_invoice_total(samples.INVOICE)
    assert result.status is ToolStatus.OK
    assert result.data == {
        "total": "1250.00",
        "currency": "USD",
        "subtotal": "1150.00",
        "tax": "100.00",
        "reconciles": True,
    }


def test_regression_subtotal_is_never_read_as_the_total():
    # A later "Sub-total" line ties with "Total" on priority, so without the
    # exclusion the most recent line (the subtotal) would win.
    result = extract_invoice_total("Total: $80.00\nSub-total: $70.00\nTax: $10.00")
    assert result.data["total"] == "80.00"
    assert result.data["subtotal"] == "70.00"


@pytest.mark.parametrize(
    "text",
    [
        "Invoice 7. Subtotal: $40.00. Tax: $5.20. Total due: $45.20",
        "Invoice 7 | Subtotal: $40.00 | Tax: $5.20 | Total due: $45.20",
        "Invoice 7; Subtotal: $40.00, Tax: $5.20, Total due: $45.20",
    ],
)
def test_single_line_invoice_pasted_from_a_pdf_is_parsed(text):
    result = extract_invoice_total(text)
    assert result.data["total"] == "45.20"
    assert result.data["subtotal"] == "40.00"
    assert result.data["reconciles"] is True


def test_subtotal_alone_is_not_a_total():
    assert extract_invoice_total("Subtotal: $70.00").status is ToolStatus.ERROR


def test_grand_total_outranks_a_plain_total():
    result = extract_invoice_total("Total: $10.00\nShipping: $2.00\nGrand total: $12.00")
    assert result.data["total"] == "12.00"


@pytest.mark.parametrize(
    ("text", "currency"),
    [
        ("Amount due: €99.90", "EUR"),
        ("TOTAL 1,500.00 CRC", "CRC"),
        ("Balance due: £5.00", "GBP"),
        ("Total: 42.00", None),
    ],
)
def test_currency_is_read_from_symbol_or_code(text, currency):
    assert extract_invoice_total(text).data["currency"] == currency


def test_totals_that_do_not_reconcile_are_flagged():
    result = extract_invoice_total("Subtotal: $100.00\nTax: $13.00\nTotal: $120.00")
    assert result.status is ToolStatus.OK
    assert result.data["reconciles"] is False
    assert any("reconcile" in action for action in result.next_actions)


def test_invoice_without_total_is_a_tool_error_with_next_action():
    result = extract_invoice_total("Invoice #9 from ACME, thanks for your business")
    assert result.status is ToolStatus.ERROR
    assert result.next_actions


# --- code snippet ------------------------------------------------------------------------


def test_code_question_language_and_error_type_are_extracted():
    result = extract_code_snippet(samples.CODE)
    assert result.status is ToolStatus.OK
    assert result.data["language"] == "python"
    assert result.data["error_type"] == "KeyError"
    assert result.data["has_traceback"] is True
    assert result.data["snippet_lines"] == 2


def test_language_is_inferred_when_the_fence_has_no_tag():
    result = extract_code_snippet("```\nconst add = (a, b) => a + b;\n```")
    assert result.data["language"] == "javascript"


def test_text_without_code_is_a_tool_error():
    assert extract_code_snippet("How are you?").status is ToolStatus.ERROR


# --- registry -----------------------------------------------------------------------------


def test_default_registry_covers_every_actionable_intent_with_a_matching_tool():
    assert set(DEFAULT_TOOLS) == {Intent.ACADEMIC_DOC, Intent.INVOICE, Intent.CODE_QUESTION}
    for intent, tool in DEFAULT_TOOLS.items():
        assert tool.intent is intent
        assert tool.run(samples.ACADEMIC).tool == tool.name
