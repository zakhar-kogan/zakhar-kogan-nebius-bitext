"""Start local Bitext services."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start local Bitext services")
    parser.add_argument("--no-mcp", action="store_true", help="skip the FastMCP server")
    parser.add_argument("--no-streamlit", action="store_true", help="skip the Streamlit UI")
    parser.add_argument("--streamlit-host", default="127.0.0.1")
    parser.add_argument("--streamlit-port", type=int, default=8501)
    parser.add_argument(
        "--mcp-transport",
        choices=("stdio", "http", "sse", "streamable-http"),
        default="http",
    )
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=8000)
    parser.add_argument("--mcp-path")
    return parser


def selected_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if not args.no_streamlit:
        commands.append(
            (
                "streamlit",
                [
                    "streamlit",
                    "run",
                    "src/bitext_agent/streamlit_app.py",
                    f"--server.address={args.streamlit_host}",
                    f"--server.port={args.streamlit_port}",
                ],
            )
        )
    if not args.no_mcp:
        command = [
            "fastmcp",
            "run",
            "src/bitext_agent/mcp_server.py:mcp",
            "--transport",
            args.mcp_transport,
        ]
        if args.mcp_transport != "stdio":
            command.extend(["--host", args.mcp_host, "--port", str(args.mcp_port)])
            if args.mcp_path:
                command.extend(["--path", args.mcp_path])
        commands.append(("mcp", command))
    return commands


def run_commands(commands: list[tuple[str, list[str]]]) -> int:
    if not commands:
        print("Nothing to start: both --no-streamlit and --no-mcp were set.", file=sys.stderr)
        return 2

    processes = [subprocess.Popen(command) for _name, command in commands]
    try:
        while True:
            for name, process in zip((name for name, _command in commands), processes):
                return_code = process.poll()
                if return_code is not None:
                    print(f"{name} exited with code {return_code}; stopping.", file=sys.stderr)
                    return return_code
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


def main(argv: list[str] | None = None) -> int:
    return run_commands(selected_commands(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
