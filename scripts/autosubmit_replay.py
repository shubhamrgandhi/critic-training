#!/usr/bin/env python3
"""Replay a trajectory's bash commands inside a fresh Singularity container,
then run the canonical agent submit command and return the resulting patch.

Used to recover patches for non-Submitted instances. The patch produced here
is byte-identical to what the agent would have gotten if it had voluntarily
submitted at the end (uses the exact same submit command).

This script is READ-ONLY with respect to the trajectory and result files.
It writes its result to a NEW preds-autosubmit.json (or returns it via stdout
when called for a single instance). Existing preds.json/report.json/traj.json
files are never modified.

Usage (single instance, prints patch JSON to stdout):
    python3 autosubmit_replay.py --traj <path-to-traj.json>

Usage (batch replay over a results dir, write preds-autosubmit.json):
    python3 autosubmit_replay.py \
        --results-dir <dir> \
        --workers 8 \
        --output preds-autosubmit.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The exact submission command string that the agent uses. Must match the
# `## Submission` line in the agent config templates. If it ever differs, the
# patches won't match — sanity-check mode catches this.
SUBMIT_CMD = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"

SUBMIT_MARKERS = {"MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}

BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)


def derive_swebench_image(instance_id: str) -> str:
    """Same logic as get_swebench_docker_image_name() in swebench.py."""
    id_docker_compatible = instance_id.replace("__", "_1776_")
    image = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return f"docker://{image}"


def extract_image_from_traj(traj: dict) -> str | None:
    """Read the image (already in docker:// form) directly from the trajectory's
    saved env config. Falls back to deriving from instance_id."""
    cfg = traj.get("info", {}).get("config", {})
    env = cfg.get("environment", {})
    img = env.get("image")
    if img:
        return img
    inst = traj.get("instance_id")
    if inst:
        return derive_swebench_image(inst)
    return None


def extract_env_settings(traj: dict) -> tuple[str, dict[str, str], int]:
    """Pull cwd, env vars, timeout from the trajectory's saved env config."""
    cfg = traj.get("info", {}).get("config", {})
    env_cfg = cfg.get("environment", {})
    cwd = env_cfg.get("cwd", "/testbed")
    env_vars = env_cfg.get("env", {}) or {}
    timeout = env_cfg.get("timeout", 60)
    return cwd, env_vars, timeout


def parse_assistant_action(content: str) -> str | None:
    """Extract the bash command from an assistant message. Returns None if
    there's not exactly 1 bash block (matches agent's parse_action behavior)."""
    matches = BASH_BLOCK_RE.findall(content)
    if len(matches) != 1:
        return None
    return matches[0].strip()


def _collect_docker_cred_sets() -> list[dict[str, str]]:
    """Collect Docker cred sets from env vars suffixed _1, _2, ... for rate-limit rotation."""
    overlays = []
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


def build_sandbox(image: str, retries: int = 10) -> Path:
    """Build a writable singularity sandbox dir from the given image, rotating
    Docker credentials across attempts to dodge per-account rate limits."""
    last_err = None
    cred_sets = _collect_docker_cred_sets()
    for attempt in range(retries):
        sandbox = Path(tempfile.gettempdir()) / f"autosubmit-{uuid.uuid4().hex[:8]}"
        run_env = os.environ.copy()
        if cred_sets:
            overlay = cred_sets[attempt % len(cred_sets)]
            run_env.update(overlay)
        try:
            subprocess.run(
                ["singularity", "build", "--sandbox", str(sandbox), image],
                check=True,
                capture_output=True,
                timeout=900,
                env=run_env,
            )
            return sandbox
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            shutil.rmtree(sandbox, ignore_errors=True)
            last_err = e
    raise RuntimeError(f"Failed to build sandbox for {image} after {retries} attempts: {last_err}")


def cleanup_sandbox(sandbox: Path) -> None:
    shutil.rmtree(sandbox, ignore_errors=True)


def exec_in_sandbox(
    sandbox: Path,
    command: str,
    cwd: str,
    env_vars: dict[str, str],
    timeout: int,
) -> dict:
    """Same singularity exec invocation as the production env (without the
    edit_obs_final_only post-processing — irrelevant for replay correctness)."""
    cmd = ["singularity", "exec", "--contain", "--cleanenv"]
    if cwd and cwd != "/":
        cmd.extend(["--pwd", cwd])
    for k, v in env_vars.items():
        cmd.extend(["--env", f"{k}={v}"])
    cmd.extend(["--writable", str(sandbox), "bash", "-c", command])
    try:
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}
    except subprocess.TimeoutExpired as e:
        return {"output": (e.stdout.decode("utf-8", "replace") if e.stdout else "") + f"\n[autosubmit: command timed out after {timeout}s]", "returncode": -1}


def extract_patch_from_submit_output(output: str) -> str:
    """Same logic as agent has_finished(): patch is everything after the marker line."""
    lines = output.lstrip().splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.strip() in SUBMIT_MARKERS:
            return "".join(lines[idx + 1:])
    # No marker found — empty patch
    return ""


def replay_and_submit(
    traj_path: Path,
    *,
    skip_last_n_steps: int = 0,
    log_progress: bool = False,
) -> dict:
    """Replay the trajectory's bash commands and run the canonical submit.

    Returns dict with keys:
      - instance_id: str
      - patch: str (the model_patch suitable for preds.json)
      - error: str | None  (set if replay or submit failed)
      - replayed_steps: int (number of bash commands actually executed)
      - skipped_steps: int (assistant msgs with !=1 bash block)
      - elapsed_sec: float
      - exit_status_original: str (the original info.exit_status)
      - submit_returncode: int (returncode of the final submit command)

    skip_last_n_steps: if you want to compare against the agent's voluntary submit
        on a Submitted run, you'd typically pass skip_last_n_steps=1 to skip the
        agent's own final submit step (so the canonical submit at the end
        produces the same patch). For sanity-check use only.
    """
    t0 = time.time()
    with open(traj_path) as f:
        traj = json.load(f)

    instance_id = traj.get("instance_id") or traj_path.parent.name
    image = extract_image_from_traj(traj)
    if not image:
        return {
            "instance_id": instance_id,
            "patch": "",
            "error": "Could not determine container image for instance",
            "replayed_steps": 0,
            "skipped_steps": 0,
            "elapsed_sec": time.time() - t0,
            "exit_status_original": traj.get("info", {}).get("exit_status"),
            "submit_returncode": None,
        }

    cwd, env_vars, timeout = extract_env_settings(traj)

    # Collect bash commands in order. Last N steps optionally skipped (sanity check).
    msgs = traj.get("messages", [])
    asst_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    if skip_last_n_steps > 0:
        asst_msgs = asst_msgs[:-skip_last_n_steps] if skip_last_n_steps < len(asst_msgs) else []

    sandbox = None
    try:
        if log_progress:
            print(f"[{instance_id}] Building sandbox from {image}", flush=True, file=sys.stderr)
        sandbox = build_sandbox(image)
        if log_progress:
            print(f"[{instance_id}] Sandbox ready at {sandbox}", flush=True, file=sys.stderr)

        replayed = 0
        skipped = 0
        for i, m in enumerate(asst_msgs):
            cmd = parse_assistant_action(m.get("content", ""))
            if cmd is None:
                skipped += 1
                continue
            # Skip if this command IS the submit command — we only want to run our
            # canonical submit at the end, not the agent's voluntary submit (whose
            # output is the patch, not a state-changing command).
            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd or "MINI_SWE_AGENT_FINAL_OUTPUT" in cmd:
                continue
            try:
                exec_in_sandbox(sandbox, cmd, cwd, env_vars, timeout)
                replayed += 1
            except Exception as e:
                # Soft-fail individual steps — match the agent's behavior, which
                # also continues even when commands fail.
                if log_progress:
                    print(f"[{instance_id}] step {i+1} replay failed (continuing): {e}", flush=True, file=sys.stderr)

            if log_progress and replayed % 25 == 0:
                print(f"[{instance_id}] replayed {replayed} steps", flush=True, file=sys.stderr)

        # Run the canonical submit. Use a generous timeout because git diff can be
        # slow on large repos.
        submit_result = exec_in_sandbox(sandbox, SUBMIT_CMD, cwd, env_vars, max(timeout * 5, 300))
        patch = extract_patch_from_submit_output(submit_result["output"])

        return {
            "instance_id": instance_id,
            "patch": patch,
            "error": None,
            "replayed_steps": replayed,
            "skipped_steps": skipped,
            "elapsed_sec": time.time() - t0,
            "exit_status_original": traj.get("info", {}).get("exit_status"),
            "submit_returncode": submit_result["returncode"],
        }
    except Exception as e:
        return {
            "instance_id": instance_id,
            "patch": "",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "replayed_steps": 0,
            "skipped_steps": 0,
            "elapsed_sec": time.time() - t0,
            "exit_status_original": traj.get("info", {}).get("exit_status"),
            "submit_returncode": None,
        }
    finally:
        if sandbox is not None:
            cleanup_sandbox(sandbox)


def find_trajectories(results_dir: Path) -> list[Path]:
    """Find every <instance>/<instance>.traj.json file under results_dir."""
    out = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        traj = entry / f"{entry.name}.traj.json"
        if traj.exists():
            out.append(traj)
    return out


def load_existing_preds(results_dir: Path) -> dict:
    p = results_dir / "preds.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def write_preds_autosubmit(
    out_path: Path,
    preds: dict,
    log_path: Path | None = None,
) -> None:
    """Write the autosubmit predictions to a NEW file. Refuses to overwrite
    preds.json (safety guard)."""
    if out_path.name == "preds.json":
        raise RuntimeError(f"Refusing to write to {out_path} — that's the original preds.json")
    out_path.write_text(json.dumps(preds, indent=2))
    if log_path:
        log_path.write_text(json.dumps({"completed_at": time.time(), "n": len(preds)}, indent=2))


def batch_replay(
    results_dir: Path,
    output_path: Path,
    workers: int = 8,
    only_non_submitted: bool = True,
    model_name: str | None = None,
    progress_log: Path | None = None,
    resume: bool = False,
) -> None:
    """Replay non-Submitted instances and write a preds-autosubmit.json.

    For Submitted instances, copies the existing patch from preds.json.
    For non-Submitted: replays trajectory, runs canonical submit, captures patch.
    """
    trajs = find_trajectories(results_dir)
    existing = load_existing_preds(results_dir)

    # Decide model_name (used in the output entries to mirror preds.json format)
    if model_name is None:
        # Pick from the first existing entry
        if existing:
            sample = next(iter(existing.values()))
            model_name = sample.get("model_name_or_path", "facebook/cwm")
        else:
            model_name = "facebook/cwm"

    out_preds: dict = {}
    todo: list[Path] = []

    # Resume mode: load any existing preds-autosubmit.json. Keep entries that
    # succeeded (non-empty patch and no _autosubmit_error). Re-run the rest.
    existing_autosubmit: dict = {}
    if resume and output_path.exists():
        try:
            existing_autosubmit = json.loads(output_path.read_text())
        except Exception:
            existing_autosubmit = {}
        print(f"Resume: {len(existing_autosubmit)} entries in existing preds-autosubmit.json", flush=True)

    for traj in trajs:
        try:
            with open(traj) as f:
                t = json.load(f)
        except Exception:
            continue
        inst = t.get("instance_id") or traj.parent.name
        es = t.get("info", {}).get("exit_status")
        if es == "Submitted" and only_non_submitted:
            # Carry forward the original submitted patch
            if inst in existing:
                out_preds[inst] = {
                    "model_name_or_path": existing[inst].get("model_name_or_path", model_name),
                    "instance_id": inst,
                    "model_patch": existing[inst].get("model_patch", ""),
                }
            else:
                # Submitted but missing from preds.json — fall back to info.submission
                out_preds[inst] = {
                    "model_name_or_path": model_name,
                    "instance_id": inst,
                    "model_patch": t.get("info", {}).get("submission", "") or "",
                }
        else:
            # Resume: skip if previous run already produced a non-empty patch
            # without error for this instance. Otherwise re-run.
            if resume:
                prev = existing_autosubmit.get(inst)
                if prev and "_autosubmit_error" not in prev and prev.get("model_patch"):
                    out_preds[inst] = prev
                    continue
            todo.append(traj)

    print(f"Total trajectories: {len(trajs)}", flush=True)
    print(f"Submitted (carried forward): {len(out_preds)}", flush=True)
    print(f"Need replay: {len(todo)}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print(f"Output: {output_path}", flush=True)

    # Periodically dump partial progress so a long run can be resumed/inspected
    completed = 0
    failed = 0
    t_start = time.time()

    def _save():
        # write atomically
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out_preds, indent=2))
        os.replace(tmp, output_path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(replay_and_submit, t): t for t in todo}
        for fut in as_completed(futs):
            traj = futs[fut]
            try:
                result = fut.result()
            except Exception as e:
                completed += 1
                failed += 1
                inst = traj.parent.name
                out_preds[inst] = {
                    "model_name_or_path": model_name,
                    "instance_id": inst,
                    "model_patch": "",
                    "_autosubmit_error": f"{type(e).__name__}: {e}",
                }
                continue
            inst = result["instance_id"]
            if result["error"]:
                failed += 1
                out_preds[inst] = {
                    "model_name_or_path": model_name,
                    "instance_id": inst,
                    "model_patch": "",
                    "_autosubmit_error": result["error"][:500],
                }
            else:
                out_preds[inst] = {
                    "model_name_or_path": model_name,
                    "instance_id": inst,
                    "model_patch": result["patch"],
                    "_autosubmit_meta": {
                        "replayed_steps": result["replayed_steps"],
                        "skipped_steps": result["skipped_steps"],
                        "elapsed_sec": round(result["elapsed_sec"], 1),
                        "original_exit_status": result["exit_status_original"],
                    },
                }
            completed += 1
            if completed % 5 == 0 or completed == len(todo):
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 1e-3)
                remaining = (len(todo) - completed) / max(rate, 1e-6)
                print(
                    f"  [{completed}/{len(todo)}] {inst}: "
                    f"patch_len={len(out_preds[inst].get('model_patch','') or '')}, "
                    f"failed={failed}, "
                    f"rate={rate:.2f}/s, eta={remaining/60:.1f}min",
                    flush=True,
                )
                _save()
                if progress_log:
                    progress_log.write_text(json.dumps({
                        "completed": completed, "total": len(todo),
                        "failed": failed,
                        "elapsed_sec": int(elapsed),
                    }))

    _save()
    print(f"Done. completed={completed}, failed={failed}, output={output_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, help="Single trajectory file to replay (prints JSON result)")
    ap.add_argument("--results-dir", type=Path, help="Results directory to batch-replay")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output preds-autosubmit.json path (default: <results-dir>/preds-autosubmit.json)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-last-n-steps", type=int, default=0,
                    help="Single-instance only: skip last N agent steps before submit (sanity check)")
    ap.add_argument("--include-submitted", action="store_true",
                    help="Also re-replay instances the agent voluntarily submitted (sanity check mode)")
    ap.add_argument("--progress-log", type=Path, default=None,
                    help="Write progress to this file periodically")
    ap.add_argument("--resume", action="store_true",
                    help="Skip instances already done (with non-empty patch and no error) in existing preds-autosubmit.json")
    args = ap.parse_args()

    if args.traj:
        result = replay_and_submit(
            args.traj,
            skip_last_n_steps=args.skip_last_n_steps,
            log_progress=True,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["error"] is None else 1

    if args.results_dir:
        out = args.output or (args.results_dir / "preds-autosubmit.json")
        if out.name == "preds.json":
            print("ERROR: refusing to write to preds.json", file=sys.stderr)
            return 2
        batch_replay(
            args.results_dir,
            out,
            workers=args.workers,
            only_non_submitted=not args.include_submitted,
            progress_log=args.progress_log,
            resume=args.resume,
        )
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
