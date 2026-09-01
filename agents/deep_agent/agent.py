"""Deployable crew & catering operations agent for Modules 3 (deploy) and 5 (Engine).

Built with deepagents' prebuilt `create_deep_agent` — a few lines give you a
planning ops agent with a subagent, skills, memory, and human-in-the-loop.

Exported as a factory `agent(config)` so a LangSmith assistant can pin runtime
config. The defaults reproduce the original behavior; the Module 5 `engine-demo`
assistant flips on a broken policy-lookup tool and weak Context Hub guidance so
Engine has a recurring issue to find.
"""

import os

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from deepagents.backends.context_hub import ContextHubBackend
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from tavily import TavilyClient

from utils.models import model
from utils.search import resilient_tavily_search

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


@tool(parse_docstring=True)
def policy_lookup(query: str) -> str:
    """Search the web for a crew, catering, dietary, or regulatory policy.

    Args:
        query: Policy or regulation to look up.
    """
    return resilient_tavily_search(query, max_retries=2)


# NOTE: kept named `easy_search` on purpose — Module 5 (Engine) and
# utils/engine.py filter traces by this exact tool name. Rename in all three
# places together if you retheme the Engine demo.
@tool(parse_docstring=True)
def easy_search(query: str) -> str:
    """Search the web for a crew, catering, dietary, or regulatory policy.

    Args:
        query: Policy or regulation to look up.
    """
    # Module 5 demo: a "lightweight" lookup that drops each result's content,
    # so the agent answers from titles alone — ungrounded by design. A fresh
    # client per call gives each concurrent tool thread its own (non-thread-safe)
    # requests.Session; a shared client races and drops connections.
    hits = TavilyClient().search(query, max_results=3).get("results", [])
    return "\n".join(f"- {h['title']} ({h['url']})" for h in hits)


def agent(config: RunnableConfig | None = None):
    """Build the agent. A LangSmith assistant can pin `config.configurable`:
    `search_tool` ("policy" default | "easy"), `interrupts` (default True), and
    `context_repo` (a Context Hub repo mounted at /context/; none by default).
    """
    cfg = (config or {}).get("configurable", {})
    lookup_tool = easy_search if cfg.get("search_tool") == "easy" else policy_lookup
    context_repo = cfg.get("context_repo")

    # `namespace` is required as of deepagents 0.7.0 (the old implicit
    # assistant_id scoping is gone); same namespace the notebooks use.
    routes = {"/memories/": StoreBackend(namespace=lambda rt: ("memories", "shared"))}
    if context_repo:
        routes["/context/"] = ContextHubBackend(context_repo)

    return create_deep_agent(
        model=model,
        tools=[lookup_tool],
        system_prompt=(
            "You are an expert crew and catering operations assistant for an airline. "
            "Help with crew scheduling and duty-time/rest compliance, in-flight catering "
            "and special-meal planning, and crew recovery during disruptions."
        ),
        memory=["./AGENTS.md"] + (["/context/AGENTS.md"] if context_repo else []),
        skills=["./skills/"],
        subagents=[{
            "name": "crew-ops-agent",
            "description": "Delegate crew and catering lookups. Give one flight or policy question at a time.",
            "system_prompt": "You are a crew and catering operations assistant. Use tools to gather "
                             "information and cite sources. Limit to 3 lookups.",
            "tools": [lookup_tool],
        }],
        backend=CompositeBackend(
            default=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
            routes=routes,
        ),
        interrupt_on={"write_file": True, "edit_file": True} if cfg.get("interrupts", True) else {},
    )
