import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("litellm_model")


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")


class LitellmModel:
    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs), timeout=1000
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        except litellm.exceptions.BadRequestError as e:
            # Some providers (notably Bedrock for Qwen) return context-window
            # exhaustion as BadRequestError instead of ContextWindowExceededError.
            # Detect by message and re-raise so the agent terminates cleanly
            # instead of retrying the same too-long prompt.
            msg = str(getattr(e, "message", "")) + " " + str(e)
            if (
                "maximum context length" in msg
                or "context length" in msg.lower()
                or "input tokens" in msg.lower() and "total of at least" in msg.lower()
                or "prompt is too long" in msg.lower()
            ):
                raise litellm.exceptions.ContextWindowExceededError(
                    message=str(e),
                    model=self.config.model_name,
                    llm_provider=self.config.model_kwargs.get("custom_llm_provider", "unknown"),
                ) from e
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        try:
            cost = litellm.cost_calculator.completion_cost(response)
        except Exception as e:
            logger.critical(
                f"Error calculating cost for model {self.config.model_name}: {e}. "
                "Please check the 'Updating the model registry' section in the documentation at "
                "https://klieret.short.gy/litellm-model-registry Still stuck? Please open a github issue for help!"
            )
            raise
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
