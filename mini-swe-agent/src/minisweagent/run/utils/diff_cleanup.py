"""Strip noise (.venv/, __pycache__/, etc.) from git-diff content embedded in strings.

R2E-Gym matplotlib containers ship with a pre-existing .venv/ in /testbed that
isn't gitignored, so `git add -A && git diff --cached` sweeps the entire venv
into the submission output. This module strips those file diff sections so we
don't persist gigabytes of noise per trajectory.

Used both at save time (auto-cleanup in save_traj) and post-hoc by scripts.
"""

from __future__ import annotations

import os
import re

# Top-level path components whose diff sections should always be stripped.
_STRIP_PATH_PATTERNS = [
    r"\.venv(/|$)",
    r"venv(/|$)",
    r"__pycache__(/|$)",
    r"\.git(/|$)",
    r"\.pytest_cache(/|$)",
    r"\.mypy_cache(/|$)",
]
_STRIP_RE = re.compile("|".join(_STRIP_PATH_PATTERNS))

_DIFF_GIT_RE = re.compile(r"(?m)^(?=diff --git )")
_DIFF_GIT_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/\S+")

# Env var to disable cleanup entirely (escape hatch — keep raw originals).
_DISABLE_ENV = "MSWEA_DISABLE_DIFF_CLEANUP"


def is_disabled() -> bool:
    return os.getenv(_DISABLE_ENV, "").lower() in ("1", "true", "yes")


def clean_diff_text(text: str) -> str:
    """Strip diff sections matching noise path patterns from a string.

    If text contains no `diff --git` headers, returns it unchanged (no-op).
    Otherwise, splits on diff section boundaries and drops sections whose
    file path matches one of the noise patterns. Preamble (text before the
    first diff header) is always preserved.
    """
    if is_disabled() or not text or "diff --git " not in text:
        return text

    parts = _DIFF_GIT_RE.split(text)
    if len(parts) <= 1:
        return text

    kept: list[str] = []
    for i, part in enumerate(parts):
        if i == 0 and not part.startswith("diff --git "):
            # Preamble (e.g. CRLF warnings) before any diff section
            if part:
                kept.append(part)
            continue
        m = _DIFF_GIT_HEADER_RE.match(part)
        if m and _STRIP_RE.search(m.group(1)):
            continue  # drop noisy file's diff section
        kept.append(part)
    return "".join(kept)


def clean_message_content(content):
    """Apply clean_diff_text to a message's content (str or list-of-parts)."""
    if isinstance(content, str):
        return clean_diff_text(content)
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                new_item = dict(item)
                for k in ("text", "value", "content"):
                    if isinstance(new_item.get(k), str):
                        new_item[k] = clean_diff_text(new_item[k])
                out.append(new_item)
            else:
                out.append(item)
        return out
    return content
