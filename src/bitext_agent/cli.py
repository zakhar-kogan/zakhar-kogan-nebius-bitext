"""Interactive command-line interface for the Bitext agent."""

from __future__ import annotations

import argparse
import sys

from bitext_agent.graph import AgentService


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Bitext customer support data analyst agent")
    parser.add_argument("--session", default="demo", help="Persistent conversation session ID")
    parser.add_argument("--user-id", default="demo", help="External user ID for profile memory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the interactive CLI loop."""

    args = build_parser().parse_args(argv)
    service = AgentService()
    print(f"Bitext agent session={args.session} user_id={args.user_id}")
    print("Type /exit to save profile memory and exit.")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            message = "/exit"
            print()
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            saved = service.distill_session(args.session, args.user_id)
            if saved:
                print("Saved profile facts:")
                for fact in saved:
                    print(f"- {fact}")
            print("Goodbye.")
            return 0
        try:
            response = service.run_turn(message, args.session, args.user_id)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        print("\nReasoning:")
        for step in response.reasoning:
            print(f"- {step.title}: {step.detail}")
        print("\nAnswer:")
        print(response.answer)
        print()


if __name__ == "__main__":
    raise SystemExit(main())

