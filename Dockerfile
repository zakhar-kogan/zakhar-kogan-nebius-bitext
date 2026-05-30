FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir uv==0.5.4
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY langgraph.json .env.example ./
RUN uv sync --frozen --no-dev
EXPOSE 8501 8000
CMD ["uv", "run", "streamlit", "run", "src/bitext_agent/streamlit_app.py", "--server.address=0.0.0.0"]
