# NAT Demo: Weather & Time Agents

A demo project for [NVIDIA NeMo Agent Toolkit (NAT)](https://docs.nvidia.com/nemo/agent-toolkit/) showcasing how to build agentic AI with custom tools and a LangGraph sub-agent.

## What's Inside

- **Weather Tool** — Get today's weather for any city (Open-Meteo API, no key needed)
- **LangGraph Time Agent** — A LangGraph `create_react_agent` registered as a NAT function with two tools:
  - `get_current_time` — Current time for any city via timezone lookup
  - `convert_time` — Convert time between cities
- **Orchestration** — NAT's built-in `react_agent` workflow routes queries to the right tool
- **Evaluation** — RAGAS-based answer accuracy eval via `nat eval`
- **Observability** — Langfuse tracing via OpenTelemetry

## Project Structure

```
src/nat_demo/
  register.py                  # Entry point — registers all functions with NAT
  weather_tool.py              # Weather tool (simple NAT function)
  langgraph_time_agent.py      # LangGraph time agent (agent-as-a-tool pattern)
  configs/config.yml           # Template config (uses env vars)
configs/config.yml             # Local runtime config (gitignored)
data/simple_eval.json          # Evaluation dataset
docker-compose.yml             # NeMo Agent UI + Langfuse stack
```

## Setup

```bash
# Create venv and install
uv venv && source .venv/bin/activate
uv pip install -e .

# Set your LLM credentials
export API_KEY="your-api-key"
export BASE_URL="your-llm-endpoint"
export MODEL_NAME="your-model-name"

# Copy template config for local use
cp src/nat_demo/configs/config.yml configs/config.yml
```

## Usage

### CLI

```bash
# Validate config
nat validate --config_file ./configs/config.yml

# Run single queries
nat run --config_file ./configs/config.yml --input "What is the current time in Tokyo?"
nat run --config_file ./configs/config.yml --input "When in Warsaw it is 15:00, what time is it in Tokyo?"
nat run --config_file ./configs/config.yml --input "What is the weather in Berlin?"
```

### API Server

```bash
nat serve --config_file ./configs/config.yml
# API available at http://localhost:8000
```

### Chat UI and Tracing

```bash
docker compose up -d
# NeMo Agent UI at http://localhost:3010
# Langfuse at http://localhost:3011
```

Login to Langfuse with the pre-configured credentials:
- **Email:** `admin@demo.com`
- **Password:** `123qweASDzxc`

### Evaluation

```bash
nat eval --config_file ./configs/config.yml
# Results in .tmp/evals/
```

## Key Patterns Demonstrated

1. **Simple Tool** (`weather_tool.py`) — Register a function with `@register_function`, yield `FunctionInfo.from_fn`
2. **Agent-as-a-Tool** (`langgraph_time_agent.py`) — Build a LangGraph agent internally, expose it as a NAT function that the outer workflow can call
3. **Config-driven orchestration** (`config.yml`) — Wire LLMs, tools, and workflows declaratively
