#!/usr/bin/env python3
"""
Compute stats for SWE-Bench Verified Mini (50 instances), organized into
three clean, readable tables:

  1. baseline_stats.csv  — no-PRM runs (base CWM, oracle, critic_selected)
  2. opus_prm_stats.csv  — Claude Opus PRM runs
  3. sft_prm_stats.csv   — Qwen 8B SFT PRM runs

Each row has explicit columns for inference_prompt, k, step_aware, post_process,
plus the SFT training config (for SFT table only).

Model costs are recomputed from token counts using full input pricing
(no prompt-caching discounts).

Usage:
    python get_stats_mini50_v2.py
    python get_stats_mini50_v2.py --parent-dir <path>
    python get_stats_mini50_v2.py --report-file report-docker.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Per-token pricing (full input, no cache discounts).
# Opus intentionally omitted so recompute_prm_cost falls through to stored cost (includes cache-write premium).
MODEL_PRICING = {
    "facebook/cwm":                                                                                 {"input": 1.8e-7, "output": 1.8e-7},
    "facebook/cwm-sft":                                                                             {"input": 1.8e-7, "output": 1.8e-7},
    "SWE-bench/SWE-agent-LM-32B":                                                                  {"input": 7.1e-8, "output": 2.83e-7},
    "Qwen/Qwen3-8B":                                                                                {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean":                            {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample":                 {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened":                        {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn":                        {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened":{"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-multiturn":{"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn":              {"input": 5e-8, "output": 1.5e-7},
}

# Training-data config for each SFT PRM (for the "train" column in sft table).
SFT_TRAIN_CONFIG = {
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean":                                  "detailed / k=5 / swebench / flattened",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample":                       "detailed / k=5 / swebench / flattened (RS)",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened":                              "concise / k=10 / swebench / flattened",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn":                              "concise / k=10 / swebench / multiturn",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened":      "concise / k=10 / r2egym / flattened",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-multiturn":      "concise / k=10 / r2egym / multiturn",
    "qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn":                    "detailed / k=5 / r2egym / multiturn",
    "qwen3-8b":                                                                            "(base model - not finetuned)",
}

# Short human-readable PRM model name for the "sft_model" column.
SFT_MODEL_NAME = {
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean":                                  "SFT v1",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample":                       "SFT v1 (RS)",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened":                              "SFT v2 flat",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn":                              "SFT v2 mt",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened":      "SFT r2e-v2 flat",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-multiturn":      "SFT r2e-v2 mt",
    "qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn":                    "SFT r2e-v1 mt",
    "qwen3-8b":                                                                            "Qwen3-8B base",
}


# Template-hash -> prompt-name.  Allows classifying runs by PRM system-prompt
# without relying on parsing the run directory name.
PROMPT_HASH_TO_NAME = {
    "674225fc2d": "detailed",
    "adf350376a": "concise",
}


# Reference run that defines the 50-instance mini subset
REFERENCE_RUN = "singularity_edit_obs_final_only_0_cwm"
BASE_CWM_DIR = REFERENCE_RUN
BASE_RUN_DIRS = [f"singularity_edit_obs_final_only_{i}_cwm" for i in range(5)]


def recompute_model_cost(traj: dict) -> float:
    config = traj.get("info", {}).get("config", {})
    model_name = config.get("model", {}).get("model_name", "")
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return traj.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0)

    total_cost = 0.0
    for m in traj.get("messages", []):
        if "extra" not in m:
            continue
        usage = m["extra"]["response"].get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_cost += input_tokens * pricing["input"] + output_tokens * pricing["output"]
    return total_cost


def recompute_prm_cost(traj: dict) -> float:
    info = traj.get("info", {})
    prm_stats = info.get("prm_stats") or {}
    stored_cost = prm_stats.get("prm_cost", 0.0)

    config = info.get("config", {})
    prm_model_name = config.get("prm_model", {}).get("model_name", "")
    pricing = MODEL_PRICING.get(prm_model_name)
    if pricing is None:
        return stored_cost

    feedback_log = prm_stats.get("prm_feedback_log", [])
    has_token_data = any(entry.get("usage") for entry in feedback_log)
    if has_token_data:
        total_cost = 0.0
        for entry in feedback_log:
            usage = entry.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_cost += input_tokens * pricing["input"] + output_tokens * pricing["output"]
        return total_cost
    return stored_cost


def get_instance_ids(run_dir: Path) -> set[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("MariusHobbhahn/swe-bench-verified-mini", split="test")
        return set(ds["instance_id"])
    except Exception:
        return {p.name for p in run_dir.iterdir() if p.is_dir()}


def classify_run(run_dir: Path) -> dict:
    """Return metadata for a run by reading its first traj.json."""
    import hashlib
    trajs = list(run_dir.glob("*/*.traj.json"))
    if not trajs:
        return {}
    traj = json.loads(trajs[0].read_text())
    ac = traj.get("info", {}).get("config", {}).get("agent", {})
    prm_cfg = traj.get("info", {}).get("config", {}).get("prm_model", {})

    use_prm = ac.get("use_prm", False)
    prm_tmpl = ac.get("prm_template", "") or ""
    tmpl_hash = hashlib.md5(prm_tmpl.encode()).hexdigest()[:10] if prm_tmpl else ""
    prompt_name = PROMPT_HASH_TO_NAME.get(tmpl_hash, tmpl_hash or "-")

    return {
        "use_prm": use_prm,
        "k": ac.get("prm_interval"),
        "prompt": prompt_name if use_prm else "-",
        "step_aware": (ac.get("step_aware_threshold") or 0) > 0,
        "post_process": bool(ac.get("prm_postprocess")),
        "prm_model_name": prm_cfg.get("model_name", "") or "",
    }


def compute_stats(run_dir: Path, instance_ids: set[str], report_filename: str = "report.json") -> dict:
    report_path = run_dir / report_filename
    resolved_ids = set()
    if report_path.exists():
        with open(report_path) as f:
            resolved_ids = set(json.load(f).get("resolved_ids", []))

    resolved_in_subset = resolved_ids & instance_ids

    exit_counts = Counter()
    total_steps = 0
    total_cost = 0.0
    total_prm_cost = 0.0
    n_found = 0

    for instance_id in sorted(instance_ids):
        inst_dir = run_dir / instance_id
        if not inst_dir.exists():
            continue
        traj_files = list(inst_dir.glob("*.traj.json"))
        if not traj_files:
            continue
        with open(traj_files[0]) as f:
            traj = json.load(f)

        info = traj.get("info", {})
        exit_counts[info.get("exit_status", "unknown")] += 1
        messages = traj.get("messages", [])
        steps = sum(
            1 for m in messages
            if m.get("role") == "assistant"
            and re.search(r'```bash\n.*?\n```', m.get("content", ""), re.DOTALL)
        )
        total_steps += steps
        total_cost += recompute_model_cost(traj)
        total_prm_cost += recompute_prm_cost(traj)
        n_found += 1

    submitted = exit_counts.get("Submitted", 0)
    n = len(instance_ids)

    return {
        "n_subset": n,
        "n_found": n_found,
        "resolved": len(resolved_in_subset),
        "res_rate_subset": 100 * len(resolved_in_subset) / n if n else 0,
        "res_rate_submitted": (100 * len(resolved_in_subset) / submitted) if submitted > 0 else None,
        "avg_steps": total_steps / n if n else 0,
        "avg_model_cost": total_cost / n if n else 0,
        "avg_prm_cost": total_prm_cost / n if n else 0,
        "avg_total_cost": (total_cost + total_prm_cost) / n if n else 0,
        "exit_counts": dict(exit_counts),
        "submitted": submitted,
    }


def load_traj(run_dir: Path, instance_id: str) -> dict | None:
    inst_dir = run_dir / instance_id
    if not inst_dir.exists():
        return None
    traj_files = list(inst_dir.glob("*.traj.json"))
    if not traj_files:
        return None
    with open(traj_files[0]) as f:
        return json.load(f)


def traj_steps(traj: dict) -> int:
    return sum(
        1 for m in traj.get("messages", [])
        if m.get("role") == "assistant"
        and re.search(r'```bash\n.*?\n```', m.get("content", ""), re.DOTALL)
    )


def compute_fallback_stats(prm_dir: Path, base_dir: Path, instance_ids: set[str], report_filename: str = "report.json") -> dict:
    prm_resolved = set()
    if (prm_dir / report_filename).exists():
        prm_resolved = set(json.load(open(prm_dir / report_filename)).get("resolved_ids", []))
    base_resolved = set()
    if (base_dir / report_filename).exists():
        base_resolved = set(json.load(open(base_dir / report_filename)).get("resolved_ids", []))

    fb_steps = 0
    fb_cost = 0.0
    fb_exit_counts = Counter()
    n_fallback = 0
    combined_resolved = 0
    prm_submitted = 0

    for instance_id in sorted(instance_ids):
        prm_traj = load_traj(prm_dir, instance_id)
        if prm_traj is None:
            continue
        prm_exit = prm_traj.get("info", {}).get("exit_status", "unknown")
        if prm_exit == "Submitted":
            prm_submitted += 1
            if instance_id in prm_resolved:
                combined_resolved += 1
        else:
            if instance_id in prm_resolved:
                combined_resolved += 1
            n_fallback += 1
            base_traj = load_traj(base_dir, instance_id)
            if base_traj is not None:
                base_exit = base_traj.get("info", {}).get("exit_status", "unknown")
                fb_exit_counts[base_exit] += 1
                fb_steps += traj_steps(base_traj)
                fb_cost += recompute_model_cost(base_traj)
                if instance_id not in prm_resolved and instance_id in base_resolved:
                    combined_resolved += 1

    fb_submitted = fb_exit_counts.get("Submitted", 0)
    return {
        "n_fallback": n_fallback,
        "fb_steps": fb_steps,
        "fb_cost": fb_cost,
        "fb_exit_counts": dict(fb_exit_counts),
        "fb_submitted": fb_submitted,
        "combined_resolved": combined_resolved,
        "combined_submitted": prm_submitted + fb_submitted,
        "n_subset": len(instance_ids),
    }


def sft_slug_from_dir(dirname: str) -> str | None:
    marker = "_cwm_prm_"
    idx = dirname.find(marker)
    if idx < 0:
        return None
    return dirname[idx + len(marker):]


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"))
    parser.add_argument("--report-file", type=str, default="report.json")
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    ref_dir = parent / REFERENCE_RUN
    if not ref_dir.exists():
        print(f"ERROR: reference run not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    instance_ids = get_instance_ids(ref_dir)
    n = len(instance_ids)
    print(f"Reference: {REFERENCE_RUN}", file=sys.stderr)
    print(f"Subset size: {n} instances\n", file=sys.stderr)

    base_dir = parent / BASE_CWM_DIR

    # Discover runs
    all_dirs = sorted([d for d in parent.iterdir() if d.is_dir()])

    # ──────────────────────────────────────────────────────────────────
    # Table 1: Baselines (no PRM)
    # ──────────────────────────────────────────────────────────────────
    baseline_headers = [
        "Run", f"Resolved (/{n})", f"Submitted (/{n})",
        f"Res. Rate (/{n})", "Res. Rate (/submitted)",
        f"Avg Steps (/{n})", f"Avg Total Cost ($) (/{n})",
    ]
    baseline_rows = []

    # Individual base runs
    for i in range(5):
        d = parent / f"singularity_edit_obs_final_only_{i}_cwm"
        if not d.exists():
            continue
        s = compute_stats(d, instance_ids, args.report_file)
        baseline_rows.append([
            f"Base CWM (run {i})",
            str(s["resolved"]), str(s["submitted"]),
            f"{s['res_rate_subset']:.2f}",
            f"{s['res_rate_submitted']:.2f}" if s['res_rate_submitted'] is not None else "N/A",
            f"{s['avg_steps']:.2f}",
            f"{s['avg_total_cost']:.3f}",
        ])

    # Base cwm-sft
    d = parent / "singularity_edit_obs_final_only_0_cwm-sft"
    if d.exists():
        s = compute_stats(d, instance_ids, args.report_file)
        baseline_rows.append([
            "Base cwm-sft (run 0)",
            str(s["resolved"]), str(s["submitted"]),
            f"{s['res_rate_subset']:.2f}",
            f"{s['res_rate_submitted']:.2f}" if s['res_rate_submitted'] is not None else "N/A",
            f"{s['avg_steps']:.2f}",
            f"{s['avg_total_cost']:.3f}",
        ])

    # Best-of-5 oracle
    bo5_resolved = set()
    total_steps_all = 0
    total_cost_all = 0.0
    n_base = 0
    bo5_submitted_iids = set()
    for base_dirname in BASE_RUN_DIRS:
        bd = parent / base_dirname
        if not bd.exists():
            continue
        n_base += 1
        rp = bd / args.report_file
        if rp.exists():
            bo5_resolved |= set(json.load(open(rp)).get("resolved_ids", [])) & instance_ids
        for iid in instance_ids:
            tr = load_traj(bd, iid)
            if tr:
                total_steps_all += traj_steps(tr)
                total_cost_all += recompute_model_cost(tr)
                if tr.get("info", {}).get("exit_status") == "Submitted":
                    bo5_submitted_iids.add(iid)
    if n_base > 0:
        bo5_res = len(bo5_resolved)
        bo5_sub = len(bo5_submitted_iids)
        avg_steps_all = total_steps_all / n
        avg_cost_all = total_cost_all / n
        baseline_rows.append([
            f"Best-of-{n_base} oracle",
            str(bo5_res), str(bo5_sub),
            f"{100 * bo5_res / n:.2f}",
            f"{100 * bo5_res / bo5_sub:.2f}" if bo5_sub else "N/A",
            f"{avg_steps_all:.2f}",
            f"{avg_cost_all:.3f}",
        ])

    # critic_selected
    critic_dir = parent / "critic_selected_cwm"
    if critic_dir.exists():
        cs = compute_stats(critic_dir, instance_ids, args.report_file)
        # Critic uses the total base-run budget
        critic_total_cost = (total_cost_all / n) if n else 0
        # Critic's own inference cost
        critic_cost = 0.0
        cr_path = critic_dir / "all_critic_results.json"
        if cr_path.exists():
            crs = json.load(open(cr_path))
            input_tok = sum(r.get("formatted_tokens", 0) for r in crs)
            output_tok = len(crs) * 500
            critic_pricing = MODEL_PRICING.get("facebook/cwm", {"input": 1.8e-7, "output": 1.8e-7})
            critic_cost = input_tok * critic_pricing["input"] + output_tok * critic_pricing["output"]
        baseline_rows.append([
            "critic_selected (CWM critic on 5 runs)",
            str(cs["resolved"]), str(cs["submitted"]),
            f"{cs['res_rate_subset']:.2f}",
            f"{cs['res_rate_submitted']:.2f}" if cs['res_rate_submitted'] is not None else "N/A",
            f"{avg_steps_all:.2f}",
            f"{critic_total_cost + critic_cost/n:.3f}",
        ])

    # ──────────────────────────────────────────────────────────────────
    # Tables 2 and 3: PRM runs (split by model)
    # ──────────────────────────────────────────────────────────────────
    opus_headers = [
        "PRM Inference Prompt", "k", "Step-aware",
        f"Resolved (/{n})", f"Submitted (/{n})",
        f"Res. Rate (/{n})", "Res. Rate (/submitted)",
        f"Avg Steps (/{n})", f"Avg Total Cost ($) (/{n})",
    ]
    sft_headers = [
        "PRM Model", "Training Data", "PRM Inference Prompt", "k", "Step-aware",
        f"Resolved (/{n})", f"Submitted (/{n})",
        f"Res. Rate (/{n})", "Res. Rate (/submitted)",
        f"Avg Steps (/{n})", f"Avg Total Cost ($) (/{n})",
    ]
    opus_rows = []
    sft_rows = []

    def fmt(stats):
        return [
            str(stats["resolved"]), str(stats["submitted"]),
            f"{stats['res_rate_subset']:.2f}",
            f"{stats['res_rate_submitted']:.2f}" if stats['res_rate_submitted'] is not None else "N/A",
            f"{stats['avg_steps']:.2f}",
            f"{stats['avg_total_cost']:.3f}",
        ]

    def fmt_fb(stats, fb):
        # stats is PRM-only; fb has the fallback numbers. Compute combined.
        combined_res = fb["combined_resolved"]
        combined_sub = fb["combined_submitted"]
        res_rate_sub = (100 * combined_res / combined_sub) if combined_sub > 0 else None
        total_steps = stats["avg_steps"] + fb["fb_steps"] / n
        total_model_cost = stats["avg_model_cost"] + fb["fb_cost"] / n
        total_prm_cost = stats["avg_prm_cost"]
        total_cost = total_model_cost + total_prm_cost
        return [
            str(combined_res), str(combined_sub),
            f"{100 * combined_res / n:.2f}",
            f"{res_rate_sub:.2f}" if res_rate_sub is not None else "N/A",
            f"{total_steps:.2f}",
            f"{total_cost:.3f}",
        ]

    # Also handle the 75-step opus reference (from a different parent dir)
    opus75_dir = Path(str(parent).replace("results_singularity_max_150_steps_prefix",
                                           "results_singularity_max_75_steps")) / \
        "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_claude-opus-4-6"
    if opus75_dir.exists():
        meta = classify_run(opus75_dir)
        s = compute_stats(opus75_dir, instance_ids, args.report_file)
        fb = compute_fallback_stats(opus75_dir, base_dir, instance_ids, args.report_file)
        opus_rows.append([
            f"{meta['prompt']} (75-step ref)",
            str(meta["k"]),
            "yes" if meta["step_aware"] else "no",
            *fmt(s),
        ])
        opus_rows.append([
            f"{meta['prompt']} (75-step ref) + base fallback",
            str(meta["k"]), "yes" if meta["step_aware"] else "no",
            *fmt_fb(s, fb),
        ])

    # Classify every PRM dir and distribute between opus_rows / sft_rows
    for d in all_dirs:
        slug = sft_slug_from_dir(d.name)
        if slug is None:
            continue  # not a PRM run
        meta = classify_run(d)
        if not meta.get("use_prm"):
            continue

        s = compute_stats(d, instance_ids, args.report_file)
        # Skip incomplete runs (<25 instances found) — not enough mini50 coverage to be meaningful
        if s["n_found"] < 25:
            continue
        fb = compute_fallback_stats(d, base_dir, instance_ids, args.report_file)

        is_opus = "claude" in slug.lower()
        if is_opus:
            opus_rows.append([
                meta["prompt"], str(meta["k"]),
                "yes" if meta["step_aware"] else "no",
                *fmt(s),
            ])
            opus_rows.append([
                meta["prompt"] + " + base fallback",
                str(meta["k"]), "yes" if meta["step_aware"] else "no",
                *fmt_fb(s, fb),
            ])
        else:
            model_name = SFT_MODEL_NAME.get(slug, slug[:25])
            train_cfg = SFT_TRAIN_CONFIG.get(slug, "?")
            sft_rows.append([
                model_name, train_cfg,
                meta["prompt"], str(meta["k"]),
                "yes" if meta["step_aware"] else "no",
                *fmt(s),
            ])
            sft_rows.append([
                model_name + " + base fallback", train_cfg,
                meta["prompt"], str(meta["k"]),
                "yes" if meta["step_aware"] else "no",
                *fmt_fb(s, fb),
            ])

    # Sort SFT rows: by training data, then by model, then by prompt/k
    def sft_sort_key(r):
        # r = [model_name, train_cfg, prompt, k, sa, pp, ...]
        is_fallback = "fallback" in r[0]
        return (r[1], r[0].replace(" + base fallback", ""), r[2], int(r[3]) if str(r[3]).isdigit() else 0, is_fallback)
    sft_rows.sort(key=sft_sort_key)

    # Sort Opus rows: by prompt, then k
    def opus_sort_key(r):
        is_fallback = "fallback" in r[0]
        prompt_key = r[0].replace(" + base fallback", "")
        return (prompt_key, int(r[1]) if str(r[1]).isdigit() else 0, is_fallback)
    opus_rows.sort(key=opus_sort_key)

    # ──────────────────────────────────────────────────────────────────
    # Write outputs
    # ──────────────────────────────────────────────────────────────────
    out_dir = parent

    def write_csv(path: Path, headers, rows):
        with open(path, "w") as f:
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join(row) + "\n")
        print(f"Saved: {path}", file=sys.stderr)

    write_csv(out_dir / "mini50_baselines.csv", baseline_headers, baseline_rows)
    write_csv(out_dir / "mini50_opus_prm.csv", opus_headers, opus_rows)
    write_csv(out_dir / "mini50_sft_prm.csv", sft_headers, sft_rows)

    # Pretty-print each table to stdout
    def print_table(title, headers, rows):
        print(f"\n{'='*80}\n{title}\n{'='*80}")
        # Column widths
        widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        print("-+-".join("-" * w for w in widths))
        for r in rows:
            print(" | ".join(r[i].ljust(widths[i]) for i in range(len(headers))))

    print_table(f"TABLE 1: Baselines (no PRM), SWE-Bench Verified Mini ({n})", baseline_headers, baseline_rows)
    print_table(f"TABLE 2: Claude Opus PRM runs, SWE-Bench Verified Mini ({n})", opus_headers, opus_rows)
    print_table(f"TABLE 3: Qwen-8B SFT PRM runs, SWE-Bench Verified Mini ({n})", sft_headers, sft_rows)


if __name__ == "__main__":
    main()
