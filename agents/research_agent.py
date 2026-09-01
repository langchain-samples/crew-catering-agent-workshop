"""Shared crew & catering operations agent used by Module 4 (LangSmith).

This is a minimal useful Deep Agent for an airline's crew and catering desk:
a crew-ops subagent + domain tools (crew roster, duty-time / rest compliance,
catering plans, disruption rebooking) + a policy web-search tool + a
checkpointer. It deliberately omits HITL and disk writes so that evaluation
runs in Module 4 don't pause or leak files.

The agent is *composed* from three pieces:

- **tools** — the domain functions defined below.
- **memory** — an `AGENTS.md` operating manual (workflow + rules).
- **skills** — `crew-briefing` / `catering-summary` output formats.

By default the manual and skills are read from the local files under
`agents/deep_agent/` (`AGENTS.md`, `skills/`). Pass `context_repo="owner/name"`
(or `"-/name"`) to instead read them from a **LangSmith Context Hub** repo
mounted at `/context/` — Module 4 pushes those files to a repo and builds the
agent against it, so the operating manual and skills are versioned in the Hub
rather than pinned to a checkout.

Module 4 imports it directly:

    from agents.research_agent import build_crew_catering_agent
    agent = build_crew_catering_agent(context_repo="-/crew-catering-context")

`build_research_agent` is kept as a backwards-compatible alias.
"""

import os
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.context_hub import ContextHubBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from utils.models import model as default_model
from utils.search import resilient_tavily_search

# Local operating manual + skills live alongside the deployable agent so the
# notebook and the deployment share one source of truth.
_AGENT_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "deep_agent")


# --------------------------------------------------------------------------- #
# Mock operational data — stands in for the crew/catering systems of record.
# In production these tools would hit the crew management system (CMS),
# the catering/provisioning platform, and the ops rebooking engine.
# --------------------------------------------------------------------------- #

_CREW_ROSTER = {
    "AC418": [
        {"name": "Capt. Diane Okafor", "role": "Captain", "duty_start": "06:10"},
        {"name": "F/O Marco Ruiz", "role": "First Officer", "duty_start": "06:10"},
        {"name": "Priya Chandran", "role": "Purser", "duty_start": "06:40"},
        {"name": "Tomas Vega", "role": "Cabin Crew", "duty_start": "06:40"},
    ],
    "AC902": [
        {"name": "Capt. Liam Fraser", "role": "Captain", "duty_start": "13:05"},
        {"name": "F/O Aisha Bello", "role": "First Officer", "duty_start": "13:05"},
        {"name": "Nina Kowalski", "role": "Purser", "duty_start": "13:35"},
    ],
}

_SPECIAL_MEALS = {
    "AC418": {"VGML": 6, "GFML": 3, "KSML": 2, "CHML": 4},  # IATA special-meal codes
    "AC902": {"VGML": 2, "DBML": 1, "MOML": 5},
}


@tool(parse_docstring=True)
def crew_roster(flight_no: str) -> str:
    """Look up the assigned crew for a flight.

    Args:
        flight_no: Flight number, e.g. "AC418".
    """
    crew = _CREW_ROSTER.get(flight_no.upper())
    if not crew:
        return f"No crew roster found for flight {flight_no!r}."
    lines = [f"Crew for {flight_no.upper()}:"]
    lines += [f"- {c['role']}: {c['name']} (duty start {c['duty_start']})" for c in crew]
    return "\n".join(lines)


@tool(parse_docstring=True)
def duty_time_check(role: str, duty_hours: float) -> str:
    """Check a crew member's planned duty period against rest/duty-time limits.

    Args:
        role: Crew role, e.g. "Captain", "First Officer", or "Cabin Crew".
        duty_hours: Planned flight-duty-period length in hours.
    """
    # Simplified flight-time-limitation (FTL) rule of thumb: 13h max FDP for
    # flight crew, 14h for cabin crew, with a 10h minimum rest after.
    limit = 14.0 if "cabin" in role.lower() or "crew" in role.lower() else 13.0
    if duty_hours > limit:
        return (
            f"NON-COMPLIANT: planned duty of {duty_hours:.1f}h exceeds the "
            f"{limit:.0f}h limit for {role}. Requires a duty reduction or standby swap; "
            "10h minimum rest still applies after the duty period."
        )
    margin = limit - duty_hours
    return (
        f"Compliant: {duty_hours:.1f}h is within the {limit:.0f}h limit for {role} "
        f"({margin:.1f}h margin). Ensure 10h minimum rest after the duty period."
    )


@tool(parse_docstring=True)
def catering_plan(flight_no: str) -> str:
    """Look up the special-meal provisioning plan for a flight.

    Args:
        flight_no: Flight number, e.g. "AC418".
    """
    meals = _SPECIAL_MEALS.get(flight_no.upper())
    if not meals:
        return f"No catering plan found for flight {flight_no!r}."
    lines = [f"Special meals loaded for {flight_no.upper()} (IATA codes):"]
    lines += [f"- {code}: {qty}" for code, qty in meals.items()]
    lines.append(f"Total special meals: {sum(meals.values())}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def rebook_crew(flight_no: str, reason: str) -> str:
    """Propose a crew recovery plan when a flight is delayed or cancelled.

    Args:
        flight_no: Affected flight number, e.g. "AC902".
        reason: Disruption reason, e.g. "3h delay" or "cancellation".
    """
    crew = _CREW_ROSTER.get(flight_no.upper())
    n = len(crew) if crew else 0
    return (
        f"Recovery plan for {flight_no.upper()} ({reason}): "
        f"hold {n} assigned crew on standby, check each against duty-time limits "
        "before re-assignment, and pull reserve crew from the base pool if any "
        "member breaches their FDP. Notify catering to re-time meal loading."
    )


@tool(parse_docstring=True)
def policy_lookup(query: str) -> str:
    """Search the web for a crew, catering, dietary, or regulatory policy.

    Args:
        query: Policy or regulation to look up.
    """
    # `resilient_tavily_search` retries on Tavily failure with linear backoff,
    # then falls back to a topic-matched canned response so a flaky network
    # doesn't break the workshop. See utils/search.py for details.
    return resilient_tavily_search(query, max_retries=2)


def build_crew_catering_agent(
    context_repo: str | None = None,
    model: BaseChatModel | str | None = None,
):
    """Return a fresh crew & catering operations deep agent.

    Args:
        context_repo: Optional LangSmith Context Hub repo (``"owner/name"`` or
            ``"-/name"``). When set, the agent reads its operating manual and
            skills from that repo (mounted at ``/context/``) instead of the
            local files under ``agents/deep_agent/``.
        model: Optional chat model (or ``init_chat_model`` string) to power the
            agent. Defaults to the workshop's shared model in ``utils/models``.
            Pass different models to compare efficacy and cost across sizes.

    Each call returns a new agent with a fresh checkpointer — useful so eval
    runs don't share state with each other.
    """
    model = model if model is not None else default_model
    tools = [crew_roster, duty_time_check, catering_plan, rebook_crew, policy_lookup]

    crew_ops_subagent = {
        "name": "crew-ops-agent",
        "description": (
            "Delegate crew and catering operations lookups. Give one flight or "
            "one policy question at a time."
        ),
        "system_prompt": (
            f"You are a crew and catering operations assistant for an airline. "
            f"Today is {datetime.now().strftime('%Y-%m-%d')}.\n"
            "Use the domain tools to check rosters, duty-time/rest compliance, "
            "catering plans, and disruption recovery. Use policy_lookup only for "
            "regulatory or dietary policy questions. Limit to 3 tool calls."
        ),
        "tools": tools,
    }

    # Compose the operating manual (memory) and output-format skills. Both are
    # read from the agent's backend, so the paths below are backend-relative.
    if context_repo:
        # Read the manual and skills from a LangSmith Context Hub repo mounted
        # at /context/. StateBackend keeps the agent's own scratch files
        # in-memory (no disk writes during eval).
        memory = ["/context/AGENTS.md"]
        skills = ["/context/skills/"]
        backend = CompositeBackend(
            default=StateBackend(),
            routes={"/context/": ContextHubBackend(context_repo)},
        )
    else:
        # Read the manual and skills from the local checkout. virtual_mode keeps
        # the agent's scratch writes off disk while still serving these files.
        memory = ["./AGENTS.md"]
        skills = ["./skills/"]
        backend = FilesystemBackend(root_dir=_AGENT_ASSETS_DIR, virtual_mode=True)

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a crew and catering operations assistant for an airline. "
            "Help with crew scheduling and duty-time/rest compliance, in-flight "
            "catering and special-meal planning, and crew recovery during "
            "disruptions. Delegate lookups to the crew-ops-agent. Follow your "
            "AGENTS.md operating manual, and use a skill when one fits the "
            "requested output format."
        ),
        subagents=[crew_ops_subagent],
        memory=memory,
        skills=skills,
        backend=backend,
        checkpointer=MemorySaver(),
    )


# Backwards-compatible alias so older imports keep working.
build_research_agent = build_crew_catering_agent
