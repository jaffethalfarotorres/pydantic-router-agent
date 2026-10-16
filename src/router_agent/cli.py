"""Command line: route one input and print the ``AgentResult`` as JSON.

router-agent "Invoice #12 ... Total due: $40.00"
echo "some text" | router-agent --backend groq
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence

from router_agent.agent import build_agent
from router_agent.providers import BACKENDS, BackendUnavailable, LLMBackend, backend_from_env

BACKEND_CHOICES = ["none", *BACKENDS]


def resolve_backend(name: str, env: Mapping[str, str] | None = None) -> LLMBackend | None:
    """``none`` means regex-only routing: no LLM, no credentials, no cost."""
    return None if name == "none" else backend_from_env(name, os.environ if env is None else env)


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="router-agent", description=__doc__.splitlines()[0])
    parser.add_argument("text", nargs="?", help="input to route; read from stdin if omitted")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="none")
    args = parser.parse_args(argv)

    try:
        backend = resolve_backend(args.backend, env)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = args.text if args.text is not None else sys.stdin.read()
    result = build_agent(backend).run(text)
    print(result.model_dump_json(indent=2))
    return 0
