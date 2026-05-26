import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minisweagent import Agent, __version__
from minisweagent.run.utils.diff_cleanup import clean_diff_text, clean_message_content


def _get_class_name_with_module(obj: Any) -> str:
    """Get the full class name with module path."""
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def _asdict(obj: Any) -> dict:
    """Convert config objects to dicts."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)  # type: ignore[arg-type]
    return obj  # let's try our luck


def save_traj(
    agent: Agent | None,
    path: Path,
    *,
    print_path: bool = True,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    print_fct: Callable = print,
    **kwargs,
):
    """Save the trajectory of the agent to a file.

    Args:
        agent: The agent to save the trajectory of.
        path: The path to save the trajectory to.
        print_path: Whether to print confirmation of path to the terminal.
        exit_status: The exit status of the agent.
        result: The result/submission of the agent.
        extra_info: Extra information to save (will be merged into the info dict).
        **kwargs: Additional information to save (will be merged into top level)

    """
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "instance_cost": 0.0,
                "api_calls": 0,
            },
            "mini_version": __version__,
        },
        "messages": [],
        "trajectory_format": "mini-swe-agent-1",
    } | kwargs
    if agent is not None:
        data["info"]["model_stats"]["instance_cost"] = agent.model.cost
        data["info"]["model_stats"]["api_calls"] = agent.model.n_calls
        data["messages"] = agent.messages
        data["info"]["config"] = {
            "agent": _asdict(agent.config),
            "model": _asdict(agent.model.config),
            "environment": _asdict(agent.env.config),
            "agent_type": _get_class_name_with_module(agent),
            "model_type": _get_class_name_with_module(agent.model),
            "environment_type": _get_class_name_with_module(agent.env),
        }
        # Save PRM stats if the agent has them
        if getattr(agent, "prm_model", None) is not None:
            # Compute total PRM tokens from feedback log
            feedback_log = getattr(agent, "prm_feedback_log", [])
            prm_input_tokens = sum(
                (entry.get("usage") or {}).get("prompt_tokens", 0)
                for entry in feedback_log
            )
            prm_output_tokens = sum(
                (entry.get("usage") or {}).get("completion_tokens", 0)
                for entry in feedback_log
            )
            data["info"]["prm_stats"] = {
                "prm_cost": getattr(agent, "prm_cost", 0.0),
                "prm_input_tokens": prm_input_tokens,
                "prm_output_tokens": prm_output_tokens,
                "prm_api_calls": agent.prm_model.n_calls,
                "prm_invocations": getattr(agent, "prm_invocations", 0),
                "prm_total_agent_steps": getattr(agent, "total_agent_steps", 0),
                "prm_feedback_history": getattr(agent, "prm_feedback_history", []),
                "prm_feedback_log": feedback_log,
                "prm_dedup_suppressions": getattr(agent, "prm_dedup_suppressions", 0),
            }
            data["info"]["prm_model_stats"] = {
                "prm_model_cost": agent.prm_model.cost,
                "prm_model_api_calls": agent.prm_model.n_calls,
            }
            data["info"]["config"]["prm_model"] = _asdict(agent.prm_model.config)
            data["info"]["config"]["prm_model_type"] = _get_class_name_with_module(agent.prm_model)
    if extra_info:
        data["info"].update(extra_info)

    # Strip noisy diff sections (.venv/, __pycache__/, etc.) from submission and
    # any message that contains a git diff. Set MSWEA_DISABLE_DIFF_CLEANUP=1 to skip.
    if isinstance(data["info"].get("submission"), str):
        data["info"]["submission"] = clean_diff_text(data["info"]["submission"])
    for msg in data.get("messages", []):
        if isinstance(msg, dict) and "content" in msg:
            msg["content"] = clean_message_content(msg["content"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    if print_path:
        print_fct(f"Saved trajectory to '{path}'")
