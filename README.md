# AI Newspaper Agent Workshop

This project is a notebook-based prototype for a personalized AI newspaper agent.

## What It Contains

- `0202_advanced_mcp_smart_tools.ipynb`: advanced newspaper MCP server with story discovery, ChromaDB-backed article memory, RAG-style context retrieval, validation, publishing, and email delivery tools.
- `0301_multi_agent_collaboration.ipynb`: Preference Agent and preference-memory MCP tools backed by ChromaDB.
- `0401_news_agent_client.ipynb`: user-facing News Agent client that connects the newspaper tools and Preference Agent.
- `0201_advanced_mcp_crud_tools.ipynb`: earlier CRUD-oriented newspaper tools.
- `01_basic_tools.ipynb`: basic tool-calling and MCP progression.
- `AGENT_PROJECT_COMBINED.md`: consolidated Chinese explanation with the core code path in one file.
- `context_overflow_manager.py`: reusable context compression and retry helpers for safer LLM calls.

## Agent Roles

The News Generator Agent is the user-facing agent. It discovers stories, creates the newspaper draft, calls the Preference Agent for review, revises if needed, validates the final draft, and publishes it.

The Preference Agent is the internal reviewer and memory manager. It searches user preference memory, returns `APPROVED` or `DENIED` feedback, and stores useful preference patterns.

## Run Order

1. Configure environment variables:

```text
OPENROUTER_API_KEY
MCP_SMTP_FROM_EMAIL
MCP_SMTP_PASSWORD
```

2. Run `0202_advanced_mcp_smart_tools.ipynb` to start the newspaper MCP server on:

```text
http://localhost:8080/mcp
```

3. Run `0301_multi_agent_collaboration.ipynb` to start the Preference Agent MCP server on:

```text
http://localhost:8082/mcp
```

4. Run `0401_news_agent_client.ipynb` and edit the final `ask_news_agent(...)` call with the user's request.

## User Interaction

Users interact with the News Generator Agent through:

```python
response = await ask_news_agent(
    "Create a short morning AI news brief for me. Review it against my preferences, revise if needed, and send it by email only after approval."
)
print(response)
```

The user does not call the Preference Agent directly. The News Generator Agent calls `preference_agent.chat(...)` internally with the complete draft content.
