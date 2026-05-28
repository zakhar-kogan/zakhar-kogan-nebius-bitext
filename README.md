# zakhar-kogan-nebius-bitext

LangGraph data analyst agent for the Bitext Customer Service dataset assignment.

The agent answers structured and unstructured questions about the dataset, declines unrelated questions, persists conversation/profile memory, exposes tools through FastMCP, and includes a Streamlit chat UI with query recommendations.

## Setup

1. Install dependencies:

   ```bash
   uv sync --extra dev
   ```

2. Create environment config:

   ```bash
   cp .env.example .env
   ```

3. Put your Nebius Token Factory key in `.env`:

   ```env
   NEBIUS_API_KEY=...
   ```

4. Download the dataset:

   ```bash
   uv run python scripts/download_dataset.py
   ```

## Models

All LLM calls use the Nebius OpenAI-compatible API.

- Router: `nvidia/Nemotron-3-Nano-Omni`, chosen for low-cost classification and query recommendation.
- Main agent: `MiniMaxAI/MiniMax-M2.5-fast`, chosen for fast tool-aware answer generation.
- Recommender: `RECOMMENDER_MODEL` defaults to the router model when unset.

The router model classifies each request before tool selection. The main model runs the ReAct loop: it chooses typed tools, receives deterministic tool observations, asks clarification when needed, and produces the final grounded answer. Dataset tools remain deterministic and bounded.

## CLI

Run an interactive session:

```bash
uv run python -m bitext_agent.cli --session demo --user-id demo
```

The CLI prints route decisions, tool calls, observations, and final answers. Profile memory is distilled during normal graph execution by default; `/exit` also performs a final distillation pass.

Try:

- `What categories exist in the dataset?`
- `How many refund requests did we get?`
- `Show me 3 examples from the REFUND category`
- `Show me 3 more`
- `Summarize how agents respond to complaint intents.`
- `What should I query next?`
- `Who is the president of France?`

## Streamlit

```bash
uv run streamlit run src/bitext_agent/streamlit_app.py
```

The UI provides chat, user-scoped session selection, a new-chat button, reasoning traces, starter recommendations, contextual recommendations, profile memory viewing/deletion, dataset diagnostics, compacted context visibility, and usage/tool-call stats.

## MCP

Start the FastMCP server:

```bash
uv run fastmcp run src/bitext_agent/mcp_server.py:mcp
```

Exposed tools include `list_categories`, `count_rows`, `show_examples`, and `intent_distribution`.

Minimal Python client shape:

```python
from fastmcp import Client

async with Client("src/bitext_agent/mcp_server.py:mcp") as client:
    result = await client.call_tool("list_categories", {})
    print(result)
```

## LangGraph Studio And LangSmith

The app works without LangSmith. For local graph debugging, set these in `.env`:

```env
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
```

`langgraph.json` points Studio at `src/bitext_agent/graph.py:graph`.

## Architecture

The graph contains these nodes:

- `load_context`
- `route_query`
- `decline_out_of_scope`
- `agent_runner`
- `query_recommendation`
- `memory_distillation`
- `finalize`

Dataset access uses Polars for dataframe operations and DuckDB for aggregation helpers. Tool specs are versioned Python modules under `src/bitext_agent/tool_specs/`; each module owns its metadata, Pydantic schemas, examples, return summary, and callable implementation.

SQLite stores users, profile facts, prompt overrides, LLM usage logs, tool-call logs, cached recommendations, pending recommendations, full conversation turns, and cached session summaries. Profile memory is distilled during graph execution by default every 3 user turns, with exact dedupe, canonical wording, and pruning to 30 active facts per user. Full session logs remain stored for assignment compliance; summaries are additional compact context for long chats, not a replacement. A separate SQLite checkpoint database stores session state keyed by session ID and is also wired into LangGraph through `SqliteSaver`.

`user_id` and `session_id` are intentionally separate. The user profile stores durable facts for a person across chats, while a session stores one resumable conversation thread. The Streamlit UI lists saved sessions for the active user by reading persisted conversation turns.

`MAX_AGENT_ITERATIONS` limits the number of ReAct model/tool cycles inside one user call and defaults to 10. `SESSION_RECENT_TURN_LIMIT` controls how many recent persisted conversation turns are included in each prompt and defaults to 6 to reduce token usage.

## Tests

```bash
uv run pytest
```

Tests cover config, dataset validation, normalization, tools, user UUID mapping, profile CRUD, prompt overrides, recommendation state, usage logging, routing, and conversation follow-ups.

## Docker

Docker is optional; local `uv` is the primary grading path.

```bash
docker compose up --build
```
