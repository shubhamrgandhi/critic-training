"""Prefix replay: reuse the first k steps from a base agent trajectory.

This allows PRM/critic experiments to share the same initial trajectory prefix
as a base (no-PRM) run, eliminating variance from vLLM sampling in the early
steps and making comparisons more controlled.

Call ``replay_prefix()`` inside the PRM agent's run() method, after the system
and instance_template messages are added but before the main step loop begins.
It replays the first ``n_steps`` bash commands from the base trajectory in the
live environment (to sync container state) and populates the agent's message
history to match.

The resulting trajectory file is indistinguishable from a fresh run — analysis
and evaluation scripts work without changes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisweagent.agents.default_prm import DefaultPRMAgent

logger = logging.getLogger(__name__)


def load_base_trajectory(base_dir: str | Path, instance_id: str) -> tuple[list[dict], str | None] | None:
    """Load the message list and submission from a base-run trajectory file.

    Returns (messages, submission) or None if the file doesn't exist.
    """
    traj_path = Path(base_dir) / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        logger.warning(f"[PREFIX] Base trajectory not found: {traj_path}")
        return None
    with open(traj_path) as f:
        data = json.load(f)
    submission = data.get("info", {}).get("submission")
    return data["messages"], submission


def replay_prefix(
    agent: DefaultPRMAgent,
    base_dir: str | Path,
    instance_id: str,
    n_steps: int,
) -> bool:
    """Replay the first *n_steps* agent steps from a base trajectory.

    Must be called **after** agent.messages already contains the system and
    instance_template messages (indices 0 and 1) but **before** the step loop.

    What it does
    ------------
    1. Loads the base trajectory for *instance_id* from *base_dir*.
    2. Validates that it has enough steps and the expected role pattern.
    3. For each of the first *n_steps* steps, extracts the bash command from
       the base assistant message and executes it in ``agent.env`` so the
       container state is correct.
    4. Appends the base trajectory's assistant and user (observation) messages
       to ``agent.messages`` — using the **base content** (not re-observed
       output) to keep the trajectory identical to the base run's prefix.
    5. Sets ``agent.model.n_calls``, ``agent.total_agent_steps``, and
       ``agent.steps_since_prm`` so the main loop and limit checks pick up
       seamlessly from step *n_steps + 1*.

    Returns True on success, False if the prefix could not be replayed (the
    caller should then run from scratch).
    """
    loaded = load_base_trajectory(base_dir, instance_id)
    if loaded is None:
        return False
    base_messages, base_submission = loaded

    # We need: system(0) + task(1) + n_steps * (assistant + user_obs)
    needed = 2 + 2 * n_steps
    if len(base_messages) < needed:
        # Base run had fewer steps than requested prefix — replay everything
        # and signal the caller to submit the base run's result directly.
        actual_steps = (len(base_messages) - 2) // 2
        if actual_steps <= 0:
            logger.warning(
                f"[PREFIX] Base trajectory for {instance_id} has no steps — running from scratch"
            )
            return False
        logger.info(
            f"[PREFIX] Base trajectory for {instance_id} has only {actual_steps} steps "
            f"(requested {n_steps}) — replaying all and submitting base result"
        )
        n_steps = actual_steps
        needed = 2 + 2 * n_steps
        agent._prefix_submit_base = True
        agent._prefix_base_submission = base_submission

    # Validate expected role pattern
    for step in range(n_steps):
        ai_idx = 2 + 2 * step
        obs_idx = ai_idx + 1
        if base_messages[ai_idx].get("role") != "assistant":
            logger.warning(
                f"[PREFIX] Expected assistant at message {ai_idx}, got "
                f"{base_messages[ai_idx].get('role')!r} — running from scratch"
            )
            return False
        if base_messages[obs_idx].get("role") != "user":
            logger.warning(
                f"[PREFIX] Expected user at message {obs_idx}, got "
                f"{base_messages[obs_idx].get('role')!r} — running from scratch"
            )
            return False

    # Replay each step: extract command using the same regex as parse_action().
    # If exactly 1 bash block → execute in sandbox (action step).
    # If 0 or 2+ bash blocks → format error in base run, nothing was executed.
    logger.info(
        f"[PREFIX] Replaying {n_steps} base-run steps for {instance_id}"
    )
    for step in range(n_steps):
        ai_idx = 2 + 2 * step
        content = base_messages[ai_idx].get("content", "")
        commands = re.findall(r"```bash\n(.*?)\n```", content, re.DOTALL)
        if len(commands) == 1:
            # Valid action — execute to sync container state
            try:
                agent.env.execute(commands[0].strip())
            except Exception as e:
                logger.warning(
                    f"[PREFIX] Environment execute failed at step {step + 1}: {e} "
                    f"— running from scratch"
                )
                return False
        else:
            # Format error in base run — no command was executed originally either
            logger.info(
                f"[PREFIX] {instance_id}: step {step + 1} was a format error "
                f"in the base run ({len(commands)} bash blocks) — no command "
                f"to execute, continuing"
            )

    # Populate agent message history with the base prefix.
    # Keep the agent's own system + task messages (already in agent.messages[0:2])
    # and append the action-observation pairs from the base trajectory.
    # For assistant messages, preserve the "extra" field (contains response metadata
    # including usage) so the saved trajectory looks identical to a fresh run.
    for i in range(2, needed):
        msg = base_messages[i]
        extra_kwargs = {}
        if msg["role"] == "assistant" and "extra" in msg:
            extra_kwargs["extra"] = msg["extra"]
        agent.add_message(msg["role"], msg.get("content", ""), **extra_kwargs)

    # Compute the cost incurred during the prefix steps from the stored token usage.
    # This ensures cost_limit checks account for the replayed prefix.
    prefix_cost = _compute_prefix_cost(base_messages, n_steps, agent.model.config.model_name)
    agent.model.cost = prefix_cost

    # Sync counters so the main loop picks up correctly:
    #   - model.n_calls: used for step_limit check in query()
    #   - total_agent_steps: PRM agent's own step counter
    #   - steps_since_prm: determines when next PRM invocation fires
    agent.model.n_calls = n_steps
    agent.total_agent_steps = n_steps
    agent.steps_since_prm = n_steps

    logger.info(
        f"[PREFIX] Replayed {n_steps} steps for {instance_id} — "
        f"continuing from step {n_steps + 1} (prefix cost: ${prefix_cost:.4f})"
    )
    return True


def _compute_prefix_cost(
    base_messages: list[dict],
    n_steps: int,
    model_name: str,
) -> float:
    """Compute the dollar cost of the first *n_steps* from stored usage data.

    Uses the litellm model registry (loaded at runtime) to get per-token rates.
    Falls back to 0.0 if cost cannot be determined (self-hosted models without
    registry entries, missing usage data, etc.).
    """
    try:
        import litellm
        model_info = litellm.model_cost.get(model_name, {})
        input_rate = model_info.get("input_cost_per_token", 0.0)
        output_rate = model_info.get("output_cost_per_token", 0.0)
    except Exception:
        return 0.0

    total = 0.0
    for step in range(n_steps):
        ai_idx = 2 + 2 * step
        usage = (
            base_messages[ai_idx]
            .get("extra", {})
            .get("response", {})
            .get("usage", {})
        )
        total += usage.get("prompt_tokens", 0) * input_rate
        total += usage.get("completion_tokens", 0) * output_rate
    return total