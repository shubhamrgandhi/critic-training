#!/usr/bin/env python3

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Set, List


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

    # Context window (lines before/after) for showing edited file snippets
    edit_diff_context: int = 5


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment with edit observation (final-only mode).
        Shows the actual file content (after changes) around edited lines,
        rather than a unified diff.
        See `SingularityEnvironmentConfig` for kwargs.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_dir = self._build_sandbox()

    @staticmethod
    def _collect_docker_cred_sets() -> list[dict[str, str]]:
        """Collect docker cred sets from env vars suffixed _1, _2, ... for rate-limit rotation.

        Reads SINGULARITY_DOCKER / APPTAINER_DOCKER / DOCKER USERNAME and PASSWORD
        for suffixes _1 through _9. Returns one overlay dict per suffix that has
        at least one credential pair. Empty list means no rotation.
        """
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
        """Execute a command in the Singularity container and return the result as a dict."""
        # --- Pre-step: track dirty files and their content ---
        pre_dirty: Set[str] = set()
        pre_content: dict[str, str] = {}
        if self._is_git_repo() and self._has_head():
            try:
                pre_dirty = self._git_dirty_set()
                # For already-dirty files, save their content to detect changes in this step
                for filepath in pre_dirty:
                    try:
                        proc = self._singularity_exec("bash", "-c", f"cat {shlex.quote(filepath)}")
                        if proc.returncode == 0:
                            pre_content[filepath] = proc.stdout
                    except Exception:
                        pass
            except Exception as e:
                self.logger.debug(f"Pre-step dirty set failed: {e}")

        # Run the user command
        cmd = [self.config.executable, "exec", "--contain", "--cleanenv"]
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

        output = result.stdout
        rc = result.returncode

        # --- Post-step: show actual file content for files that changed ---
        if rc == 0 and self._is_git_repo() and self._has_head():
            try:
                post_dirty = self._git_dirty_set()
                newly_dirty = post_dirty - pre_dirty

                # Check which already-dirty files had their content change in this step
                changed_from_pre = set()
                for filepath in pre_dirty & post_dirty:
                    if filepath in pre_content:
                        try:
                            proc = self._singularity_exec("bash", "-c", f"cat {shlex.quote(filepath)}")
                            if proc.returncode == 0 and proc.stdout != pre_content[filepath]:
                                changed_from_pre.add(filepath)
                        except Exception:
                            pass

                # Show files that are newly dirty OR had content change in this step
                targets = sorted(newly_dirty | changed_from_pre)

                if targets:
                    snippet_text = self._show_file_snippets(targets, context=self.config.edit_diff_context)
                    if snippet_text.strip():
                        output = f"{output.rstrip()}\n\nCommand Run:\n\n{command.rstrip()}\n\nFiles edited after running command (showing {self.config.edit_diff_context} lines of context):\n\n{snippet_text.rstrip()}\n"
            except Exception as e:
                self.logger.debug(f"Post-step snippet display failed: {e}")

        return {"output": output, "returncode": rc}

    def cleanup(self):
        if hasattr(self, "sandbox_dir"):
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()

    # -----------------------
    # Singularity exec helper
    # -----------------------
    def _singularity_exec(self, *args: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        cmd = [
            self.config.executable, "exec",
            "--contain", "--cleanenv",
        ]
        work_dir = self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])
        cmd.extend(["--writable", str(self.sandbox_dir), *args])
        return subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # -----------------------
    # Git helpers
    # -----------------------
    def _is_git_repo(self) -> bool:
        proc = self._singularity_exec("bash", "-c", "git rev-parse --is-inside-work-tree >/dev/null 2>&1 && echo yes || echo no")
        return proc.stdout.strip().endswith("yes")

    def _has_head(self) -> bool:
        proc = self._singularity_exec("bash", "-c", "git rev-parse --verify HEAD >/dev/null 2>&1 && echo yes || echo no")
        return proc.stdout.strip().endswith("yes")

    def _git_dirty_set(self) -> Set[str]:
        """Files currently different from HEAD (staged or unstaged)."""
        cmd = (
            "set -e; "
            "{ git diff --name-only; git diff --name-only --cached; } | sort -u"
        )
        proc = self._singularity_exec("bash", "-c", cmd)
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}

    def _git_diff_paths(self, paths: List[str], *, context: int) -> str:
        """Return one unified diff (-U<context>) for the given paths vs HEAD."""
        if not paths:
            return ""
        n = max(0, int(context))
        pathlist = " ".join(shlex.quote(p) for p in paths)
        proc = self._singularity_exec("bash", "-c", f"git diff --no-color -U{n} -- {pathlist}")
        return proc.stdout

    def _parse_diff_line_ranges(self, diff_text: str) -> dict[str, List[tuple[int, int]]]:
        """
        Parse a unified diff and extract the line ranges that were modified in the new file.
        Returns a dict mapping filename -> list of (start_line, end_line) tuples.
        """
        ranges_by_file = {}
        current_file = None

        for line in diff_text.splitlines():
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3][2:] if parts[3].startswith('b/') else parts[3]
                    ranges_by_file[current_file] = []

            elif line.startswith('@@') and current_file:
                match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    new_start = int(match.group(1))
                    new_count = int(match.group(2)) if match.group(2) else 1
                    new_end = new_start + new_count - 1
                    ranges_by_file[current_file].append((new_start, new_end))

        return ranges_by_file

    def _show_file_snippets(self, paths: List[str], context: int) -> str:
        """
        Show the actual file content (after changes) for the given paths,
        with a context window around the changed lines.
        """
        if not paths:
            return ""

        # Get the diff with minimal context to identify changed line ranges
        diff_text = self._git_diff_paths(paths, context=0)
        ranges_by_file = self._parse_diff_line_ranges(diff_text)

        output_parts = []

        for path in sorted(ranges_by_file.keys()):
            ranges = ranges_by_file[path]
            if not ranges:
                continue

            # Read the actual file content (current state after edit)
            proc = self._singularity_exec("bash", "-c", f"cat {shlex.quote(path)}")
            if proc.returncode != 0:
                continue

            lines = proc.stdout.splitlines()
            if not lines:
                continue

            # Merge overlapping ranges with context
            merged_ranges = []
            for start, end in sorted(ranges):
                range_start = max(1, start - context)
                range_end = min(len(lines), end + context)

                if merged_ranges and range_start <= merged_ranges[-1][1] + 1:
                    merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], range_end))
                else:
                    merged_ranges.append((range_start, range_end))

            # Build output for this file
            file_output = [f"=== File: {path} ==="]
            for range_start, range_end in merged_ranges:
                if len(merged_ranges) > 1:
                    file_output.append(f"\n--- Lines {range_start}-{range_end} ---")
                for i in range(range_start - 1, range_end):
                    if i < len(lines):
                        file_output.append(f"{i + 1:4d} | {lines[i]}")

            output_parts.append("\n".join(file_output))

        return "\n\n".join(output_parts)
