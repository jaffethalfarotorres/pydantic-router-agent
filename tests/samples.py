"""Realistic inputs shared by the router, tool and agent tests (all synthetic)."""

INVOICE = """ACME Hosting LLC
Invoice #2026-118
Bill to: Example Corp
Subtotal: $1,150.00
Tax: $100.00
Total due: $1,250.00
Payment terms: Net 30
"""

ACADEMIC = """CS-101 Introduction to Programming
Assignment 3: Recursion
Due date: 2026-11-20
Submit a single .py file. Worth 20 points.
"""

CODE = """Why does this fail?
```python
data = {}
print(data["missing"])
```
Traceback (most recent call last):
KeyError: 'missing'
"""

NO_SIGNAL = "Can you take a look at this when you have a minute?"
"""No deterministic signal at all: the regex layer must abstain."""

WEAK_SIGNAL = "Please send me the invoice."
"""A single keyword is not enough evidence for the regex layer."""

CONFLICT = "The invoice total looks wrong because my Python script has a bug."
"""Signals for two intents with a small margin: the regex layer must abstain."""
