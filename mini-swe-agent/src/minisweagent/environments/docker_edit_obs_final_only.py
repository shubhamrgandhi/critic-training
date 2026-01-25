import logging
import os
import shlex
import subprocess
import uuid
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Set, List


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""

    # Context window (lines before/after) for showing edited file snippets
    edit_diff_context: int = 5


class DockerEnvironment:
    def __init__(self, *, config_class: type = DockerEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        # --- Pre-step: track dirty files and their content ---
        pre_dirty: Set[str] = set()
        pre_content: dict[str, str] = {}
        if self._is_git_repo() and self._has_head():
            try:
                pre_dirty = self._git_dirty_set()
                # For already-dirty files, save their content to detect changes in this step
                for filepath in pre_dirty:
                    try:
                        proc = self._docker_exec("bash", "-lc", f"cat {shlex.quote(filepath)}")
                        if proc.returncode == 0:
                            pre_content[filepath] = proc.stdout
                    except Exception:
                        pass
            except Exception as e:
                self.logger.debug(f"Pre-step dirty set failed: {e}")

        # Run the user command
        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-lc", command])

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
                            proc = self._docker_exec("bash", "-lc", f"cat {shlex.quote(filepath)}")
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
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

    # -----------------------
    # Git helpers
    # -----------------------
    def _docker_exec(self, *args: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        assert self.container_id, "Container not started"
        cmd = [self.config.executable, "exec", "-w", self.config.cwd, self.container_id, *args]
        return subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _is_git_repo(self) -> bool:
        proc = self._docker_exec("bash", "-lc", "git rev-parse --is-inside-work-tree >/dev/null 2>&1 && echo yes || echo no")
        return proc.stdout.strip().endswith("yes")

    def _has_head(self) -> bool:
        proc = self._docker_exec("bash", "-lc", "git rev-parse --verify HEAD >/dev/null 2>&1 && echo yes || echo no")
        return proc.stdout.strip().endswith("yes")

    def _git_dirty_set(self) -> Set[str]:
        """Files currently different from HEAD (staged or unstaged)."""
        cmd = (
            "set -e; "
            "{ git diff --name-only; git diff --name-only --cached; } | sort -u"
        )
        proc = self._docker_exec("bash", "-lc", cmd)
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}

    def _git_diff_paths(self, paths: List[str], *, context: int) -> str:
        """Return one unified diff (-U<context>) for the given paths vs HEAD."""
        if not paths:
            return ""
        n = max(0, int(context))
        pathlist = " ".join(shlex.quote(p) for p in paths)
        proc = self._docker_exec("bash", "-lc", f"git diff --no-color -U{n} -- {pathlist}")
        return proc.stdout

    def _parse_diff_line_ranges(self, diff_text: str) -> dict[str, List[tuple[int, int]]]:
        """
        Parse a unified diff and extract the line ranges that were modified in the new file.
        Returns a dict mapping filename -> list of (start_line, end_line) tuples.
        """
        ranges_by_file = {}
        current_file = None
        
        for line in diff_text.splitlines():
            # Match file headers: diff --git a/path b/path
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    # Extract b/path (the new file)
                    current_file = parts[3][2:] if parts[3].startswith('b/') else parts[3]
                    ranges_by_file[current_file] = []
            
            # Match hunk headers: @@ -old_start,old_count +new_start,new_count @@
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
            proc = self._docker_exec("bash", "-lc", f"cat {shlex.quote(path)}")
            if proc.returncode != 0:
                continue
            
            lines = proc.stdout.splitlines()
            if not lines:
                continue
            
            # Merge overlapping ranges with context
            merged_ranges = []
            for start, end in sorted(ranges):
                # Add context window
                range_start = max(1, start - context)
                range_end = min(len(lines), end + context)
                
                # Merge with previous range if overlapping or adjacent
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