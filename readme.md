# ThriveAI

A documentation Q&A assistant built on LangGraph, with persistent memory, live web retrieval, and a Streamlit interface. Ask questions about your docs and get grounded, context-aware answers instead of one-shot lookups.

## Features

- **Conversational Q&A over documentation** — powered by a LangGraph agent instead of a simple retrieval-then-answer chain, so it can reason over multi-step queries.
- **Dual PostgreSQL-backed memory** — short-term (per-session) and long-term (cross-session) memory, so the assistant retains context across a conversation and across visits.
- **Live web retrieval via Exa MCP** — pulls current information through the Model Context Protocol instead of relying solely on a static, pre-indexed document store.
- **LangSmith instrumentation** — full pipeline tracing for debugging response quality and latency issues.
- **Streamlit interface** — simple web UI for interacting with the assistant.

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph |
| Retrieval | Exa MCP |
| Memory | PostgreSQL (dual-store: short-term + long-term) |
| Observability | LangSmith |
| Interface | Streamlit |
| Language | Python |

## Project Structure

```
ThriveAI/
├── MCP/                  # MCP server/client configuration for Exa retrieval
├── systemPrompt.txt      # System prompt injected into the agent
├── requirements.txt      # Python dependencies
├── test.py                # [Add a short description of what this tests]
└── .env                   # Environment variables (not committed — see below)
```

## Setup

### Prerequisites

- Python 3.10+
- A running PostgreSQL instance
- An [Exa](https://exa.ai/) API key (for MCP-based retrieval)
- A LangSmith API key (optional, for tracing)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd ThriveAI

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (never commit this file) with:

```
# [Fill in your actual variable names, e.g.]
OPENAI_API_KEY=your_key_here
EXA_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/thriveai
LANGSMITH_API_KEY=your_key_here
```

A `.env.example` with placeholder values (no real keys) is recommended so collaborators know what's required — see `.env.example` if present.

### Running the App

```bash
streamlit run app.py   # [replace with your actual entry point, e.g. main.py]
```

## Known Issues / In Progress

- Narrowing project scope before adding further features (see roadmap below).
- Recently resolved: system prompt not being injected into the agent, MCP client being recreated on every request instead of reused, and database setup running redundantly per message.

## Roadmap

- [ ] Ship the current core pipeline incrementally before expanding scope
- [ ] [Add next planned feature]

## License

[Add your license, e.g. MIT]