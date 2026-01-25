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

    # Always on: show a unified diff for files touched in this step (fast path)
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

        # --- Pre-step: capture dirty set & guess candidate paths from the command ---
        pre_dirty: Set[str] = set()
        guessed: Set[str] = set()
        if self._is_git_repo() and self._has_head():
            try:
                pre_dirty = self._git_dirty_set()
            except Exception as e:
                self.logger.debug(f"Pre-step dirty set failed: {e}")
        try:
            guessed = self._guess_candidate_paths(command)
            if guessed:
                guessed = {self._to_repo_relative(p) for p in guessed if p}
        except Exception as e:
            self.logger.debug(f"Command path guess failed: {e}")

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

        # --- Post-step: print one unified diff for files touched in this step ---
        if rc == 0 and self._is_git_repo() and self._has_head():
            try:
                post_dirty = self._git_dirty_set()
                newly_dirty = post_dirty - pre_dirty
                guessed_dirty = post_dirty.intersection(guessed)
                targets = sorted(newly_dirty.union(guessed_dirty))
                if targets:
                    diff_text = self._git_diff_paths(targets, context=self.config.edit_diff_context)
                    if diff_text.strip():
                        output = f"{output.rstrip()}\n\nCommand Run:\n\n{command.rstrip()}\n\nDiff of files edited after running command:\n\n{diff_text.rstrip()}\n"
            except Exception as e:
                self.logger.debug(f"Post-step diff failed: {e}")

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

    # -----------------------
    # Command filename parsing (sed -i, redirs, tee, perl -i)
    # -----------------------
    def _guess_candidate_paths(self, command: str) -> Set[str]:
        """
        Extract filenames likely edited by this step.
        Handles multiple subcommands chained with &&, ||, ; .
        Robust for sed -i with scripts like: 935s/.../.../, s/.../.../
        """
        cands: Set[str] = set()
        parts = re.split(r"\s*(?:&&|\|\||;)\s*", command)

        redir_re = re.compile(r'(?:^|\s)(?:>>?|2>>?|&>>?)\s*(?P<f>[^\s;&|]+)')
        for part in parts:
            # Redirections: > file, >> file, 2> file, &> file
            for m in redir_re.finditer(part):
                f = _strip_shell_quotes(m.group("f"))
                if f and f != "/dev/null":
                    cands.add(f)

            # Tokenize subcommand
            try:
                tokens = shlex.split(part)
            except Exception:
                tokens = part.split()
            if not tokens:
                continue

            # Find every 'sed' occurrence and parse its args
            i = 0
            while i < len(tokens):
                if tokens[i] != "sed":
                    i += 1
                    continue
                i += 1
                # consume options (including -i, -iEXT, -e SCRIPT, -f FILE)
                while i < len(tokens) and tokens[i].startswith("-"):
                    t = tokens[i]
                    if t == "-i":
                        # optional extension argument (may be '', or .bak). If next token is an option, skip arg.
                        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                            # treat as extension; consume but we don't use it
                            i += 1
                    # -iEXT combined, -eSCRIPT combined, -fFILE combined are fine; no separate arg to consume
                    if t == "-e" or t == "-f":
                        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                            i += 1  # consume the script/file token
                    i += 1

                # after options: one script token may appear (e.g., 's/.../.../'), then file(s)
                # skip a single script-like token if present
                if i < len(tokens) and _looks_like_sed_script(tokens[i]):
                    i += 1

                # remaining tokens until next 'sed' are filenames
                while i < len(tokens) and tokens[i] != "sed":
                    t = tokens[i]
                    # stop at pipes; we only handle simple cases for speed
                    if t in {"|"}:
                        break
                    if t and t != "/dev/null":
                        cands.add(t)
                    i += 1

        # Normalize: strip quotes, make absolute paths repo-relative, drop ./ prefix
        normed: Set[str] = set()
        for p in cands:
            p = _strip_shell_quotes(p)
            p = self._to_repo_relative(p)
            if p.startswith("./"):
                p = p[2:]
            if p:
                normed.add(p)
        return normed

    def _to_repo_relative(self, path: str) -> str:
        """Convert absolute path under cwd to repo-relative; otherwise return as-is."""
        if not path:
            return path
        try:
            base = self.config.cwd.rstrip("/") + "/"
            if path.startswith(base):
                return path[len(base):]
        except Exception:
            pass
        return path


def _strip_shell_quotes(s: str) -> str:
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1]
    return s


# Recognize sed script-like tokens so we don't mistake them for filenames.
# Examples: s/foo/bar/, 935s/x/y/, 1,10s/a/b/g
_SED_SCRIPT_RE = re.compile(r"^(?:\d+(?:,\d+)?)?\s*s[^A-Za-z0-9]\S.*")
def _looks_like_sed_script(token: str) -> bool:
    t = token.strip("\"'")
    return bool(_SED_SCRIPT_RE.match(t))
