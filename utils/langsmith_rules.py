"""Helpers for managing LangSmith run rules from code.

LangSmith supports two related automations on a tracing project:

1. **Online evaluators** — score each new run with an LLM-as-judge and attach
   the score as feedback.
2. **Annotation queue automations** — route runs matching a filter into a
   queue for a human to review.

Both are configured as "rules" against a project. The LangSmith Python SDK
doesn't expose a high-level API for them, so we call the REST endpoint
directly. `create_run_rule` returns a deep link to the rule's page in the UI.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union
from uuid import UUID

import requests
from langsmith import Client


# --------------------------------------------------------------------------- #
# Annotation queues
# --------------------------------------------------------------------------- #

def get_or_create_annotation_queue(
    client: Client,
    name: str,
    description: str = "",
    *,
    rubric_instructions: Optional[str] = None,
    rubric_items: Optional[list] = None,
):
    """Return an existing annotation queue by name, or create one.

    A bare queue has no reviewer guidance and nothing to fill in. Pass:

    - ``rubric_instructions``: guidance shown at the top of the queue.
    - ``rubric_items``: the feedback keys reviewers score, each a dict with at
      least ``feedback_key`` (see ``AnnotationQueueRubricItem``). Any referenced
      feedback key that lacks a workspace feedback config is created as a 0-1
      continuous score so the reviewer form is well-defined.

    If the queue already exists, its rubric is updated in place so re-running the
    cell keeps the instructions and feedback items in sync.
    """
    if rubric_items:
        _ensure_feedback_configs(client, rubric_items)

    existing = list(client.list_annotation_queues(name=name))
    if existing:
        queue = existing[0]
        if rubric_instructions is not None or rubric_items is not None:
            client.update_annotation_queue(
                queue.id,
                rubric_instructions=rubric_instructions,
                rubric_items=rubric_items,
            )
        return queue

    return client.create_annotation_queue(
        name=name,
        description=description,
        rubric_instructions=rubric_instructions,
        rubric_items=rubric_items,
    )


def _ensure_feedback_configs(client: Client, rubric_items: list) -> None:
    """Create a default 0-1 continuous feedback config for any rubric key missing one.

    A rubric item links a queue to a *feedback key*; the key's type/range comes
    from its workspace feedback config. If a key has no config yet, the reviewer
    form is underspecified, so we create a simple 0 (fail) - 1 (pass) score.
    """
    for item in rubric_items:
        key = item.get("feedback_key") if isinstance(item, dict) else None
        if not key:
            continue
        try:
            client.create_feedback_config(
                feedback_key=key,
                feedback_config={"type": "continuous", "min": 0, "max": 1},
            )
        except Exception:
            # Already exists (409) or otherwise present -- leave the existing config.
            pass


# --------------------------------------------------------------------------- #
# Run rules
# --------------------------------------------------------------------------- #

def _raise_with_body(response: requests.Response) -> None:
    """Like `raise_for_status()`, but include the server's response body.

    The LangSmith API returns the actual reason for a 4xx (e.g. an invalid
    evaluator config or unknown secret name) in the response body. The bare
    `raise_for_status()` hides it, leaving only a generic status line, so we
    attach the body to the error message.
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = (response.text or "").strip()
        raise requests.HTTPError(
            f"{exc} — response body: {body[:1000]}", response=response
        ) from exc


def _llm_judge_evaluator(
    prompt: Union[str, Sequence[tuple[str, str]]],
    output_schema: dict,
    *,
    model_name: str = "gpt-5.6-terra",
    temperature: float = 0,
    input_var: str = "input",
    output_var: str = "output",
) -> dict:
    """Build the `evaluators[]` payload for an LLM-as-judge online evaluator.

    `prompt` may be either:
      - a system prompt string (we'll add a default user template), or
      - a list of (role, content) tuples (full message list).

    `output_schema` is a JSON Schema dict with `title`, `description`,
    `properties`, and `required`. The LangSmith API requires those four fields.
    """
    if isinstance(prompt, str):
        messages = [
            ("system", prompt),
            ("human", f"Input: {{{{{input_var}}}}}\n\nOutput: {{{{{output_var}}}}}"),
        ]
    else:
        messages = list(prompt)

    # The judge runs on LangSmith's infrastructure, not in this process, so it never
    # sees your .env. Its credentials come from a secret stored in LangSmith →
    # Workspace settings → Secrets, referenced here by name only.
    #
    # Which secret depends on how .env is set up: LANGSMITH_API_KEY_GATEWAY present
    # means this machine talks to models through the LangSmith Gateway, so point the
    # judge there too (the gateway-only model names like `gpt-5.6-luna` require it).
    # Otherwise call OpenAI directly. We only check whether the variable is set —
    # its value stays local and is never sent; the string below is a secret *name*.
    model_kwargs = {"model": model_name, "temperature": temperature}
    if os.environ.get("LANGSMITH_API_KEY_GATEWAY"):
        model_kwargs["base_url"] = "https://gateway.smith.langchain.com/openai"
        model_kwargs["api_key"] = {"lc": 1, "type": "secret", "id": ["LANGSMITH_API_KEY_GATEWAY"]}
    else:
        model_kwargs["api_key"] = {"lc": 1, "type": "secret", "id": ["OPENAI_API_KEY"]}

    return {
        "structured": {
            "prompt": [list(m) for m in messages],
            "model": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                "kwargs": model_kwargs,
            },
            "variable_mapping": {input_var: input_var, output_var: output_var},
            "schema": output_schema,
        }
    }


def create_run_rule(
    client: Client,
    *,
    project_name: str,
    display_name: str,
    filter: str = "",
    sampling_rate: float = 1.0,
    # If set: attach an LLM-as-judge online evaluator.
    llm_judge_prompt: Optional[Union[str, Sequence[tuple[str, str]]]] = None,
    llm_judge_schema: Optional[dict] = None,
    llm_judge_model: str = "gpt-5.6-luna",
    # If set: route matching runs to this annotation queue.
    add_to_annotation_queue_id: Optional[Union[str, UUID]] = None,
) -> dict:
    """Create or replace a run rule on a tracing project.

    Returns a dict with `id`, `url` (deep link to the rule in the UI), and the
    raw `payload` LangSmith stored. Either `llm_judge_prompt` (+schema), or
    `add_to_annotation_queue_id`, or both, should be provided.
    """
    print("Creating run rule")
    project = client.read_project(project_name=project_name)

    evaluators = []
    if llm_judge_prompt is not None:
        if llm_judge_schema is None:
            raise ValueError("llm_judge_schema is required when llm_judge_prompt is set")
        evaluators.append(
            _llm_judge_evaluator(
                llm_judge_prompt, llm_judge_schema, model_name=llm_judge_model,
            )
        )

    body = {
        "display_name": display_name,
        "session_id": str(project.id),
        "sampling_rate": sampling_rate,
        "filter": filter,
        "evaluators": evaluators,
    }
    if add_to_annotation_queue_id is not None:
        body["add_to_annotation_queue_id"] = str(add_to_annotation_queue_id)

    headers = {
        "x-api-key": client.api_key,
        "content-type": "application/json",
        "accept": "application/json",
    }

    # Idempotency: if a rule with this display_name already exists in the project,
    # delete it first. POST /runs/rules has no upsert semantics, so without this
    # rerunning a notebook cell accumulates duplicate rules.
    list_response = requests.get(
        f"{client.api_url}/runs/rules",
        params={"session_id": str(project.id)},
        headers={"x-api-key": client.api_key, "accept": "application/json"},
        timeout=30,
    )
    _raise_with_body(list_response)
    for existing in list_response.json():
        if existing.get("display_name") == display_name:
            requests.delete(
                f"{client.api_url}/runs/rules/{existing['id']}",
                headers={"x-api-key": client.api_key, "accept": "application/json"},
                timeout=15,
            )

    response = requests.post(
        f"{client.api_url}/runs/rules", json=body, headers=headers, timeout=30,
    )
    _raise_with_body(response)
    payload = response.json()

    tenant_id = payload["tenant_id"]
    rule_id = payload["id"]
    evaluator_id = payload.get("evaluator_id")

    # Evaluator rules (LLM-as-judge) and automation rules (queue routing, etc.)
    # have different UI pages in LangSmith.
    if evaluator_id:
        url = (
            f"https://smith.langchain.com/o/{tenant_id}/evaluators/{evaluator_id}"
            f"?ruleId={rule_id}&sourceKind=session&sourceId={project.id}"
        )
    else:
        url = (
            f"https://smith.langchain.com/o/{tenant_id}/projects/p/{project.id}"
            f"?runview=threads&tab=2"
        )

    return {"id": rule_id, "url": url, "payload": payload}


def delete_run_rule(client: Client, rule_id: Union[str, UUID]) -> None:
    """Delete a run rule by id."""
    headers = {"x-api-key": client.api_key, "accept": "application/json"}
    response = requests.delete(
        f"{client.api_url}/runs/rules/{rule_id}", headers=headers, timeout=15,
    )
    response.raise_for_status()
