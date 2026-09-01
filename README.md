# Crew & Catering Agent Workshop

This repository contains hands-on tutorials for learning LangChain, LangGraph, and Deep Agents, built around a **crew & catering operations agent** for an airline — crew scheduling and duty-time/rest compliance, in-flight catering and special-meal planning, and crew recovery during disruptions.

This is a condensed version of LangChain Academy, intended to be run in a session with a LangChain engineer. If you're interested in going into more depth, or working through tutorials on your own, check out [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)! LangChain Academy has helpful pre-recorded videos from our LangChain engineers.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment variables
cp .env.example .env
# Edit .env and fill in your keys
```

| Key | Required for | Get one |
|-----|--------------|---------|
| `OPENAI_API_KEY` | Modules 4 & 5 (default model) | <https://platform.openai.com> |
| `LANGSMITH_API_KEY` | Modules 4 & 5 (recommended for all) | <https://smith.langchain.com> |
| `LANGSMITH_API_KEY_GATEWAY` / `WORKSPACE_ID` | LangSmith Gateway policies | same key as `LANGSMITH_API_KEY`; workspace ID from LangSmith Settings → Workspace |
| `TAVILY_API_KEY` | `policy_lookup` web-search tool | <https://tavily.com> |

```bash
# 3. Start Jupyter
uv run jupyter notebook
```

Open whichever module(s) your recipe calls for.

## Switching Models

All modules import `model` from `utils/models.py`. Change one line there to swap providers — no notebook edits required.

```python
# utils/models.py

# OpenAI (default)
model = init_chat_model("openai:gpt-5.6-terra", use_responses_api=True)

# Anthropic
# model = init_chat_model("anthropic:claude-sonnet-4-5")

# Azure OpenAI
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-5.6-terra", streaming=True)

# AWS Bedrock
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(provider="anthropic", model_id="...")
```

`utils/models.py` also ships a commented-out **LangSmith Gateway** block. Module 3 §1.4 walks through flipping the default to it so every model call (notebooks *and* the deployed agent) is routed through the gateway and subject to workspace policies.

## Engine (Module 2)

Module 2 introduces **LangSmith Engine** — it reads your deployed agent's production traces, clusters recurring failures into issues, diagnoses the root cause against your connected source code, and proposes fixes as GitHub PRs. It runs on the Module 3 deployment, driven through an *assistant* (a saved graph configuration) that swaps in a deliberately broken search tool so Engine has a clear, reproducible issue to find.

## Project Structure

```
modular-workshops/
├── README.md                       (this file — recipes + setup)
├── pyproject.toml                  (shared dependencies)
├── .env.example
├── langgraph.json                  (registers agents/deep_agent for langgraph dev)
├── utils/
├── agents/
│   ├── research_agent.py           (shared crew & catering agent factory — Module 4 imports for eval)
│   └── deep_agent/                 (deployable + governed crew & catering agent for Module 3)
│       ├── agent.py
│       ├── AGENTS.md
│       └── skills/
│           ├── crew-briefing/SKILL.md
│           └── catering-summary/SKILL.md
├── images/                         (diagrams used by the notebooks)
└── modules/
    ├── 01_langsmith.ipynb          (Module 4 — introduces the crew & catering agent, then tracing + evals)
    └── 02_engine.ipynb             (Module 5)
```

## Common Issues

**`langgraph deploy` fails with 403 / permission denied**
Your API key is a personal token. Generate a service key (`lsv2_sk_...`) in LangSmith settings.

**Notebook can't find `utils` / `agents`**
Each module's setup cell prepends `project_root` (the workshop root) to `sys.path`. If you moved a notebook, update the `Path().resolve().parent` line to point at the workshop root.

## For LangChain Internal Users
Please refer to this linked [Notion document](https://app.notion.com/p/Modular-Workshops-37d808527b1780318063fd210446aa03?source=copy_link) for instructions on setup and usage.
