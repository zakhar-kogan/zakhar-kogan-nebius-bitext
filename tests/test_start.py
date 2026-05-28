from __future__ import annotations

import argparse

from bitext_agent import start


def parse_args(argv: list[str]) -> argparse.Namespace:
    return start.build_parser().parse_args(argv)


def test_selected_commands_start_both_services_by_default() -> None:
    args = parse_args([])

    commands = start.selected_commands(args)

    assert commands == [
        (
            "streamlit",
            [
                "streamlit",
                "run",
                "src/bitext_agent/streamlit_app.py",
                "--server.address=127.0.0.1",
                "--server.port=8501",
            ],
        ),
        (
            "mcp",
            [
                "fastmcp",
                "run",
                "src/bitext_agent/mcp_server.py:mcp",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
        ),
    ]


def test_selected_commands_can_skip_mcp_and_customize_streamlit() -> None:
    args = parse_args(
        ["--no-mcp", "--streamlit-host", "0.0.0.0", "--streamlit-port", "8600"]
    )

    commands = start.selected_commands(args)

    assert commands == [
        (
            "streamlit",
            [
                "streamlit",
                "run",
                "src/bitext_agent/streamlit_app.py",
                "--server.address=0.0.0.0",
                "--server.port=8600",
            ],
        )
    ]


def test_selected_commands_can_skip_streamlit_and_customize_mcp() -> None:
    args = parse_args(
        [
            "--no-streamlit",
            "--mcp-transport",
            "sse",
            "--mcp-host",
            "0.0.0.0",
            "--mcp-port",
            "9000",
            "--mcp-path",
            "/events/",
        ]
    )

    commands = start.selected_commands(args)

    assert commands == [
        (
            "mcp",
            [
                "fastmcp",
                "run",
                "src/bitext_agent/mcp_server.py:mcp",
                "--transport",
                "sse",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--path",
                "/events/",
            ],
        )
    ]


def test_stdio_mcp_omits_network_options() -> None:
    args = parse_args(
        ["--no-streamlit", "--mcp-transport", "stdio", "--mcp-path", "/ignored/"]
    )

    assert start.selected_commands(args) == [
        (
            "mcp",
            [
                "fastmcp",
                "run",
                "src/bitext_agent/mcp_server.py:mcp",
                "--transport",
                "stdio",
            ],
        )
    ]


def test_main_returns_error_when_nothing_selected() -> None:
    assert start.main(["--no-streamlit", "--no-mcp"]) == 2
