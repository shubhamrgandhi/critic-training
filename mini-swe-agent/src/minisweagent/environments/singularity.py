#!/usr/bin/env python3

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SingularityEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    """Path to the singularity executable."""
    sandbox_build_retries: int = 10
    """Number of retries for building the sandbox if an error occurs."""


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_dir = self._build_sandbox()

    @staticmethod
    def _collect_docker_cred_sets() -> list[dict[str, str]]:
        """Collect docker cred sets from env vars suffixed _1, _2, ... for rate-limit rotation."""
        overlays: list[dict[str, str]] = []
        for i in range(1, 10):
            overlay = {}
            for prefix in ("SINGULARITY_DOCKER", "APPTAINER_DOCKER", "DOCKER"):
                user = os.getenv(f"{prefix}_USERNAME_{i}")
                pw = os.getenv(f"{prefix}_PASSWORD_{i}")
                if user and pw:
                    overlay[f"{prefix}_USERNAME"] = user
                    overlay[f"{prefix}_PASSWORD"] = pw
            if overlay:
                overlays.append(overlay)
        return overlays

    def _build_sandbox(self) -> Path:
        # Building the sandbox can fail (Docker Hub rate limits, transient
        # network), so we retry, rotating Docker creds across attempts to
        # spread load over multiple accounts.
        max_retries = self.config.sandbox_build_retries
        cred_sets = self._collect_docker_cred_sets()
        for attempt in range(max_retries):
            sandbox_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
            run_env = os.environ.copy()
            if cred_sets:
                overlay = cred_sets[attempt % len(cred_sets)]
                run_env.update(overlay)
            try:
                subprocess.run(
                    [self.config.executable, "build", "--sandbox", sandbox_dir, self.config.image],
                    check=True,
                    capture_output=True,
                    env=run_env,
                )
                break
            except subprocess.CalledProcessError as e:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                acct = run_env.get("DOCKER_USERNAME", "?") if cred_sets else "default"
                self.logger.error(
                    f"Error building image {self.config.image} (account {acct!r}), stdout: {e.stdout}, stderr: {e.stderr} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
        return sandbox_dir

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        cmd = [self.config.executable, "exec"]

        # Do not inherit directories and env vars from host
        cmd.extend(["--contain", "--cleanenv"])

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
