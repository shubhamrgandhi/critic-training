"""PRM-enhanced agent class for tool overuse reduction.

Extends DefaultAgent with a Process Reward Model (PRM) supervisor that
periodically reviews the agent's trajectory and provides feedback to
reduce redundant tool usage.  The PRM can use a different model from the
agent itself (configured via the ``prm_model`` YAML section).
"""

import re
import time
from collections.abc import Callable

from minisweagent import Environment, Model
from minisweagent.agents.default import (
    AgentConfig,
    DefaultAgent,
    NonTerminatingException,
    TerminatingException,
)
from jinja2 import StrictUndefined, Template


class DefaultPRMAgent(DefaultAgent):
    """DefaultAgent extended with periodic PRM (supervisor) feedback for tool overuse reduction.

    Key design:
    - Every ``prm_interval`` agent steps, the PRM is invoked with the full trajectory.
    - ALL supervisor messages are kept in ``self.messages`` for trajectory logging,
      tagged with ``_supervisor=True`` and ``_supervisor_active=True/False``.
    - Only the ACTIVE supervisor message is sent to the agent model; inactive ones
      are filtered out in ``query()``.
    - The active supervisor message contains a condensed summary of past feedbacks
      plus the current feedback in full.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        prm_model: Model | None = None,
        config_class: Callable = AgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.prm_model = prm_model  # Separate model for PRM queries
        self.steps_since_prm = 0
        self.total_agent_steps = 0
        self.prm_cost = 0.0
        self.prm_invocations = 0
        self.prm_feedback_history: list[dict] = []  # Running history of past feedbacks (truncated, for agent messages)
        self.prm_feedback_log: list[dict] = []  # Full untruncated log of every feedback (for trajectory/analysis)
        self.supervisor_message_index: int | None = None  # Index of current supervisor msg
        self.last_delivered_feedback: str | None = None  # Last feedback actually sent to agent (for dedup)
        self.prm_dedup_suppressions: int = 0  # Count of suppressed feedbacks

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished, with periodic PRM invocations."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        # Prefix replay: reuse first prm_interval steps from a base run
        self._prefix_submit_base = False
        if self.config.prefix_trajectory_dir:
            from minisweagent.agents.prefix_replay import replay_prefix
            instance_id = getattr(self, "instance_id", "")
            if replay_prefix(self, self.config.prefix_trajectory_dir, instance_id, self.config.prm_interval):
                # Base run had fewer steps than prm_interval — just submit its result
                if self._prefix_submit_base:
                    result = getattr(self, "_prefix_base_submission", "") or ""
                    return "Submitted", result
                # Invoke PRM on the replayed prefix immediately, matching the
                # normal flow where PRM fires after prm_interval steps.
                if self.config.use_prm:
                    try:
                        self.invoke_prm()
                    except Exception as e:
                        print(f"[PRM] ERROR: PRM unavailable after prefix replay, "
                              f"continuing without feedback: {e}")
                    self.steps_since_prm = 0

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

            self.steps_since_prm += 1
            self.total_agent_steps += 1

            # Check if we need to invoke PRM after this step
            if self.config.use_prm and self.steps_since_prm >= self.config.prm_interval:
                try:
                    self.invoke_prm()
                except Exception as e:
                    print(f"[PRM] ERROR: PRM unavailable at step {self.total_agent_steps}, "
                          f"continuing without feedback: {e}")
                self.steps_since_prm = 0

    def query(self) -> dict:
        """Query the model, expanding the active supervisor message with history context."""
        all_messages = self.messages
        # Build filtered list: drop inactive supervisor msgs, expand active one with history
        filtered = []
        for msg in all_messages:
            if msg.get("_supervisor") and not msg.get("_supervisor_active"):
                continue  # Drop inactive supervisor messages
            if msg.get("_supervisor") and msg.get("_supervisor_active"):
                # Expand with history for the agent
                filtered.append({
                    "role": msg["role"],
                    "content": self._build_supervisor_message(msg["content"]),
                })
            else:
                filtered.append(msg)
        self.messages = filtered
        try:
            response = super().query()
            new_response_msg = self.messages[-1]
        finally:
            self.messages = all_messages
        self.messages.append(new_response_msg)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and append step-aware nudge if past threshold."""
        output = super().get_observation(response)
        threshold = self.config.step_aware_threshold
        # total_agent_steps hasn't been incremented yet for this step
        current_step = self.total_agent_steps + 1
        if threshold > 0 and current_step >= threshold and self.config.step_aware_agent_template:
            nudge = Template(self.config.step_aware_agent_template, undefined=StrictUndefined).render(
                current_step=current_step, step_limit=self.config.step_limit,
            )
            self.messages[-1]["content"] += "\n\n" + nudge
        return output

    def invoke_prm(self, max_retries: int = 10, base_delay: float = 2.0):
        """Invoke the PRM supervisor with retries and exponential backoff.

        Non-retryable errors (e.g. context-window overflow on the same input)
        are raised immediately rather than retried, since retries cannot fix
        a too-long prompt.
        """
        if not self.config.prm_template:
            return

        for attempt in range(1, max_retries + 1):
            try:
                self._invoke_prm_inner()
                return
            except Exception as e:
                if self._is_prm_error_non_retryable(e):
                    print(f"[PRM] ERROR: non-retryable PRM error (attempt {attempt}): {e}")
                    raise
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"[PRM] WARNING: PRM call failed (attempt {attempt}/{max_retries}): {e}")
                    print(f"[PRM] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"[PRM] ERROR: PRM call failed after {max_retries} attempts: {e}")
                    raise

    @staticmethod
    def _is_prm_error_non_retryable(e: Exception) -> bool:
        """Errors that won't be fixed by retrying the same input."""
        try:
            import litellm
            if isinstance(e, litellm.exceptions.ContextWindowExceededError):
                return True
        except Exception:
            pass
        msg = str(e).lower()
        return (
            "contextwindowexceeded" in msg
            or "context length" in msg
            or "max_tokens" in msg and "too large" in msg
            or "prompt is too long" in msg
        )

    # Maximum number of previous feedbacks to include in PRM context
    MAX_PRM_FEEDBACK_HISTORY = 5

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _is_feedback_duplicate(self, new_feedback: str) -> tuple[bool, float]:
        """Check if new feedback is too similar to the last delivered feedback.

        Returns (is_duplicate, similarity_score). On error, returns (False, 0.0)
        so feedback is delivered rather than suppressed.
        """
        if not self.config.prm_dedup or self.last_delivered_feedback is None:
            return False, 0.0
        try:
            import litellm
            response = litellm.embedding(
                model=self.config.prm_dedup_model,
                input=[self.last_delivered_feedback, new_feedback],
            )
            emb_old = response.data[0]["embedding"]
            emb_new = response.data[1]["embedding"]
            similarity = self._cosine_similarity(emb_old, emb_new)
            return similarity >= self.config.prm_dedup_threshold, similarity
        except Exception as e:
            print(f"[PRM] WARNING: Dedup embedding call failed, delivering feedback: {e}")
            return False, 0.0

    # Tag constants for flattened format — must match training data in
    # prepare_prm_sft_data_opus_distill_full_feedback_history.py.
    _TAG_ENVIRONMENT = "[USER/ENVIRONMENT]"
    _TAG_AGENT = "[AGENT RESPONSE]"
    _TAG_FEEDBACK_PREFIX = "[PREVIOUS_FEEDBACK"  # followed by " #N]:"

    def _build_prm_messages_flattened(self, prm_system_prompt: str) -> list[dict]:
        """Build PRM query as a single flattened user message with tags."""
        parts = []

        task_context = self.messages[1]["content"]
        parts.append(f"{self._TAG_ENVIRONMENT}: {task_context}")

        for msg in self.messages[2:]:
            if msg.get("_supervisor"):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if role == "assistant":
                parts.append(f"{self._TAG_AGENT}: {content}")
            else:
                parts.append(f"{self._TAG_ENVIRONMENT}: {content}")

        if self.prm_feedback_history:
            recent = self.prm_feedback_history[-self.MAX_PRM_FEEDBACK_HISTORY:]
            for j, fb in enumerate(recent, 1):
                parts.append(f"{self._TAG_FEEDBACK_PREFIX} #{j}]: {fb['feedback']}")

        threshold = self.config.step_aware_threshold
        if threshold > 0 and self.total_agent_steps >= threshold and self.config.step_aware_prm_template:
            prm_nudge = Template(self.config.step_aware_prm_template, undefined=StrictUndefined).render(
                current_step=self.total_agent_steps, step_limit=self.config.step_limit,
            )
            parts.append(f"[CRITICAL CONTEXT]: {prm_nudge}")

        parts.append(f"{self._TAG_ENVIRONMENT}: Please provide your supervisor feedback now based on the trajectory above.")

        return [
            {"role": "system", "content": prm_system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _build_prm_messages_multiturn(self, prm_system_prompt: str) -> list[dict]:
        """Build PRM query as real multi-turn chat messages."""
        prm_messages = [
            {"role": "system", "content": prm_system_prompt},
            {"role": "user", "content": self.messages[1]["content"]},  # task context
        ]

        for msg in self.messages[2:]:
            if msg.get("_supervisor"):
                continue
            content = msg.get("content", "") or ""
            prm_messages.append({"role": msg["role"], "content": content})

        # Append feedback history + final prompt as the last user message
        final_parts = []
        if self.prm_feedback_history:
            recent = self.prm_feedback_history[-self.MAX_PRM_FEEDBACK_HISTORY:]
            final_parts.append("## Previous Supervisor Feedback History:\n")
            for j, fb in enumerate(recent, 1):
                final_parts.append(
                    f"### Feedback #{j} (after step {fb['after_step']}):\n"
                    f"{fb['feedback']}\n"
                )

        threshold = self.config.step_aware_threshold
        if threshold > 0 and self.total_agent_steps >= threshold and self.config.step_aware_prm_template:
            prm_nudge = Template(self.config.step_aware_prm_template, undefined=StrictUndefined).render(
                current_step=self.total_agent_steps, step_limit=self.config.step_limit,
            )
            final_parts.append(f"CRITICAL CONTEXT: {prm_nudge}\n")

        final_parts.append("Please provide your supervisor feedback now based on the trajectory above.")
        prm_messages.append({"role": "user", "content": "\n\n".join(final_parts)})

        # Merge consecutive same-role messages
        merged = [prm_messages[0]]
        for msg in prm_messages[1:]:
            if msg["role"] == merged[-1]["role"] and msg["role"] != "system":
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)

        return merged

    def _resolve_prm_format(self) -> str:
        """Resolve the PRM input format, auto-detecting from model name if needed."""
        fmt = self.config.prm_format
        if fmt in ("flattened", "multiturn"):
            return fmt
        # Auto-detect from PRM model name
        query_model = self.prm_model if self.prm_model else self.model
        model_name = getattr(query_model, "config", None)
        if model_name is not None:
            model_name = getattr(model_name, "model_name", "") or ""
        else:
            model_name = ""
        if "flattened" in model_name.lower():
            return "flattened"
        return "multiturn"

    def _invoke_prm_inner(self):
        """Inner PRM invocation — separated so invoke_prm can catch all exceptions."""
        prm_system_prompt = self.render_template(self.config.prm_template)

        prm_format = self._resolve_prm_format()
        if self.prm_invocations == 0:
            print(f"[PRM] Using '{prm_format}' format "
                  f"(config.prm_format='{self.config.prm_format}')")
        if prm_format == "multiturn":
            prm_messages = self._build_prm_messages_multiturn(prm_system_prompt)
        else:
            prm_messages = self._build_prm_messages_flattened(prm_system_prompt)

        # Query PRM (with best-of-N selection if configured)
        n_candidates = self.config.prm_n_candidates
        if n_candidates > 1:
            candidates = self._query_prm_candidates(prm_messages, n_candidates)
            # Strip think tags from all candidates
            candidates = [
                {**c, "content": self._strip_think_tags(c.get("content", ""))}
                for c in candidates
            ]
            if self.config.prm_select == "shortest":
                selected = min(candidates, key=lambda c: len(c.get("content", "")))
            else:
                selected = candidates[0]
            prm_response = selected
            # Log candidate lengths for analysis
            candidate_lengths = [len(c.get("content", "")) for c in candidates]
            print(f"[PRM] Best-of-{n_candidates} candidate lengths: {candidate_lengths}, "
                  f"selected: {len(prm_response.get('content', ''))} chars")
            feedback = prm_response.get("content", "")
        else:
            prm_response = self._query_prm(prm_messages)
            feedback = self._strip_think_tags(prm_response.get("content", ""))

        # Post-process feedback to remove concrete actions if enabled
        original_feedback = feedback
        feedback = self._postprocess_feedback(feedback)

        # Check for duplicate feedback before delivering
        is_duplicate, similarity = self._is_feedback_duplicate(feedback)

        # Log full feedback for trajectory (never truncated)
        log_entry = {
            "after_step": self.total_agent_steps,
            "invocation": self.prm_invocations + 1,
            "feedback": feedback,
            "original_feedback": original_feedback if feedback != original_feedback else None,
            "suppressed": is_duplicate,
            "dedup_similarity": round(similarity, 4) if similarity > 0 else None,
            "usage": prm_response.get("_prm_usage"),
        }
        if n_candidates > 1:
            log_entry["n_candidates"] = n_candidates
            log_entry["candidate_lengths"] = [len(c.get("content", "")) for c in candidates]
            log_entry["select_strategy"] = self.config.prm_select
        self.prm_feedback_log.append(log_entry)

        if is_duplicate:
            print(f"[PRM] Feedback after step {self.total_agent_steps} SUPPRESSED "
                  f"(similarity {similarity:.4f} >= threshold {self.config.prm_dedup_threshold})")
            self.prm_dedup_suppressions += 1
            self.prm_invocations += 1
            return

        if similarity > 0:
            print(f"[PRM] Feedback after step {self.total_agent_steps} DELIVERED "
                  f"(similarity {similarity:.4f} < threshold {self.config.prm_dedup_threshold})")

        # Mark old supervisor message as inactive (kept in messages for trajectory logging)
        if self.supervisor_message_index is not None:
            old_msg = self.messages[self.supervisor_message_index]
            old_msg["_supervisor_active"] = False
            self.prm_feedback_history.append({
                "after_step": self.total_agent_steps - self.config.prm_interval,
                "feedback": old_msg["content"],
            })

        # Store just the current feedback as content (clean traj logging);
        # query() expands it with history context on-the-fly for the agent.
        self.add_message(
            "user", feedback,
            _supervisor=True, _supervisor_active=True,
            _prm_after_step=self.total_agent_steps,
        )
        self.supervisor_message_index = len(self.messages) - 1
        self.last_delivered_feedback = feedback

        self.prm_invocations += 1

    def _build_supervisor_message(self, current_feedback: str) -> str:
        """Build the supervisor message with truncated history + current feedback."""
        parts = ["SUPERVISOR FEEDBACK:"]

        if self.prm_feedback_history:
            parts.append("\n## Previous Feedback (truncated):")
            for j, fb in enumerate(self.prm_feedback_history, 1):
                truncated = fb["feedback"][:1500]
                if len(fb["feedback"]) > 1500:
                    truncated += "..."
                parts.append(f"\n### Feedback #{j} (after step {fb['after_step']}):\n{truncated}")

        parts.append(f"\n## Current Feedback (after step {self.total_agent_steps}):")
        parts.append(current_feedback)

        return "\n".join(parts)

    _POSTPROCESS_PROMPT = (
        "You are revising the OVERALL_GUIDANCE section of feedback from a coding agent supervisor.\n\n"
        "The original guidance may contain concrete actions: bash commands, code edits, "
        "specific file paths, or step-by-step instructions.\n\n"
        "Rewrite it so that:\n"
        "1. Concrete actions (commands, code, file paths, step-by-step instructions) are "
        "either REMOVED or SUMMARIZED into a brief high-level strategy.\n"
        "2. The guidance tells the agent WHAT goal to pursue, not HOW to execute it.\n"
        "3. Be concise — 2-3 sentences max.\n\n"
        "Return ONLY the revised guidance text, nothing else."
    )

    def _postprocess_feedback(self, feedback: str) -> str:
        """Post-process PRM feedback: strip rubric, keep only rewritten OVERALL_GUIDANCE.

        The full rubric (SPECIFICATION/REASONING/COORDINATION ERRORS, EVIDENCE,
        RECOVERY_ACTION) is chain-of-thought scaffolding for the PRM — the agent
        only needs the high-level guidance.  The unprocessed feedback is still
        preserved in ``original_feedback`` in the log entry.
        """
        if not self.config.prm_postprocess:
            return feedback
        # Extract the OVERALL_GUIDANCE section
        match = re.search(r'OVERALL_GUIDANCE:\s*(.*)', feedback, re.DOTALL)
        if not match:
            return feedback
        original_guidance = match.group(1).strip()
        if not original_guidance:
            return feedback
        try:
            import litellm
            response = litellm.completion(
                model=self.config.prm_postprocess_model,
                messages=[
                    {"role": "system", "content": self._POSTPROCESS_PROMPT},
                    {"role": "user", "content": original_guidance},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            revised_guidance = response.choices[0].message.content or original_guidance
            print(f"[PRM] Post-processed OVERALL_GUIDANCE: {len(original_guidance)} -> {len(revised_guidance)} chars")
            return revised_guidance
        except Exception as e:
            print(f"[PRM] WARNING: Post-processing failed, using original feedback: {e}")
            return feedback

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from model output."""
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return stripped

    def _query_prm(self, messages: list[dict]) -> dict:
        """Query the PRM model for feedback and track costs separately."""
        query_model = self.prm_model if self.prm_model else self.model
        cost_before = query_model.cost

        # Clean messages for Mistral/Devstral if needed
        model_name = (
            getattr(query_model, "model_name", None)
            or getattr(query_model, "name", None)
            or (
                query_model.get_template_vars().get("model_name")
                if hasattr(query_model, "get_template_vars")
                else ""
            )
            or str(query_model)
        )

        send_messages = messages
        if any(keyword in str(model_name).lower() for keyword in ["mistral", "devstral"]):
            allowed = {"role", "content", "name", "tool_calls", "tool_call_id"}
            cleaned = []
            for m in messages:
                m2 = {k: v for k, v in m.items() if k in allowed}
                if "content" not in m2 or m2["content"] is None:
                    m2["content"] = ""
                cleaned.append(m2)
            send_messages = cleaned

        if "mistral" in str(model_name).lower():
            response = query_model.query(send_messages, stop=["</s>"])
        else:
            response = query_model.query(send_messages)

        # Track PRM-specific cost
        cost_after = query_model.cost
        self.prm_cost += cost_after - cost_before

        # Extract token usage for cost recomputation
        usage = response.get("extra", {}).get("response", {}).get("usage", {})
        response["_prm_usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        }

        return response

    def _query_prm_candidates(self, messages: list[dict], n: int) -> list[dict]:
        """Query the PRM model for N candidate responses and return all of them.

        Uses litellm.completion directly with n=N to get multiple completions
        in a single API call. Requires temperature > 0 for diversity.
        """
        import litellm as _litellm

        query_model = self.prm_model if self.prm_model else self.model
        cost_before = query_model.cost

        # Build kwargs: start from model config, override n
        kwargs = dict(query_model.config.model_kwargs)
        kwargs["n"] = n
        # Ensure temperature > 0 for diverse candidates
        if kwargs.get("temperature", 0) == 0:
            kwargs["temperature"] = 0.7

        response = _litellm.completion(
            model=query_model.config.model_name,
            messages=messages,
            **kwargs,
            timeout=1000,
        )

        # Track cost
        try:
            cost = _litellm.cost_calculator.completion_cost(response)
        except Exception:
            cost = 0.0
        query_model.cost += cost
        query_model.n_calls += 1
        self.prm_cost += cost

        from minisweagent.models import GLOBAL_MODEL_STATS
        GLOBAL_MODEL_STATS.add(cost)

        # Extract all choices
        usage = getattr(response, "usage", None)
        if usage:
            usage = dict(usage)
        else:
            usage = {}

        shared_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        }

        candidates = []
        for choice in response.choices:
            candidates.append({
                "content": choice.message.content or "",
                "_prm_usage": shared_usage,
            })

        return candidates
