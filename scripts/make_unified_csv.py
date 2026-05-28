#!/usr/bin/env python3
"""
Generate two unified CSV files covering all settings — one for mini50, one for
full500 — ready to paste into Google Sheets.

Naming convention (matches plots):
  Training Config: <source>-<teacher_prompt>-k<K>-<format>
    e.g. "r2e-detailed-k5-mt", "sb-concise-k10-flat"
  Inference Config: <inference_prompt>-k<K>[-SA]
    e.g. "detailed-k5", "concise-k10-SA"

Columns:
  Category              (Baseline / Opus Critic / SFT Critic)
  Critic Model          (short name — Opus, or SFT training-config tuple)
  Training Data         (same info expanded as "source / teacher_prompt / k / format")
  Inference Prompt      (detailed / concise / -)
  k                     (critic intervention interval at inference)
  Step-aware            (yes / no / -)
  Fallback              ("no" or "+ base fallback")
  Resolved
  Submitted
  Res. Rate             (resolved / n * 100)
  Res. Rate (/submitted)
  Avg Steps
  Avg Total Cost ($)

Usage:
    python make_unified_csv.py
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"


# Full input pricing (no cache discount). Opus omitted → stored cost used.
MODEL_PRICING = {
    "facebook/cwm":                                                                                 {"input": 1.8e-7, "output": 1.8e-7},
    "facebook/cwm-sft":                                                                             {"input": 1.8e-7, "output": 1.8e-7},
    "Qwen/Qwen3-8B":                                                                                {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean":                            {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample":                 {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened":                        {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn":                        {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened":{"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-multiturn":{"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn":              {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-instructions-k5-opus-distill-32k-lr5e6-multiturn": {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn":     {"input": 5e-8, "output": 1.5e-7},
    "shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn": {"input": 5e-8, "output": 1.5e-7},
}


# SFT slug -> short training-config tuple (matches plots)
SFT_NAME = {
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean":                                  "sb-detailed-k5-flat",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample":                       "sb-detailed-k5-flat (RS)",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-flattened":                              "sb-concise-k10-flat",
    "qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6-multiturn":                              "sb-concise-k10-mt",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-flattened":      "r2e-concise-k10-flat",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k10-opus-distill-32k-lr5e6-multiturn":      "r2e-concise-k10-mt",
    "qwen3-8b-full-sft-prm-r2egym-k5-opus-distill-32k-lr5e6-multiturn":                    "r2e-detailed-k5-mt",
    "qwen3-8b-full-sft-prm-r2egym-instructions-k5-opus-distill-32k-lr5e6-multiturn":       "r2e-concise-k5-mt",
    "qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn":           "r2esb-detailed-k5-mt",
    "qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn": "r2esb-concise-k5-mt",
    "qwen3-8b":                                                                            "Qwen3-8B (base)",
}

# Same tuple split into its fields (for the Training Data column)
SFT_TRAIN_DESC = {
    "sb-detailed-k5-flat":       "SWE-Bench / detailed / k=5 / flat",
    "sb-detailed-k5-flat (RS)":  "SWE-Bench / detailed / k=5 / flat (RS)",
    "sb-concise-k10-flat":       "SWE-Bench / concise / k=10 / flat",
    "sb-concise-k10-mt":         "SWE-Bench / concise / k=10 / mt",
    "r2e-concise-k10-flat":      "R2E-Gym / concise / k=10 / flat",
    "r2e-concise-k10-mt":        "R2E-Gym / concise / k=10 / mt",
    "r2e-detailed-k5-mt":        "R2E-Gym / detailed / k=5 / mt",
    "r2e-concise-k5-mt":         "R2E-Gym / concise / k=5 / mt",
    "r2esb-detailed-k5-mt":      "R2E-Gym+SWE-Smith / detailed / k=5 / mt",
    "r2esb-concise-k5-mt":       "R2E-Gym+SWE-Smith / concise / k=5 / mt",
    "Qwen3-8B (base)":           "(base model, not finetuned)",
}

PROMPT_HASH = {"674225fc2d": "detailed", "adf350376a": "concise"}

REFERENCE_RUN = "singularity_edit_obs_final_only_0_cwm"
BASE_RUN_DIRS = [f"singularity_edit_obs_final_only_{i}_cwm" for i in range(5)]


def recompute_model_cost(traj):
    config = traj.get("info", {}).get("config", {})
    model_name = config.get("model", {}).get("model_name", "")
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return traj.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0)
    total = 0.0
    for m in traj.get("messages", []):
        if "extra" not in m: continue
        usage = m["extra"]["response"].get("usage", {})
        total += usage.get("prompt_tokens", 0) * pricing["input"] + usage.get("completion_tokens", 0) * pricing["output"]
    return total


def recompute_prm_cost(traj):
    info = traj.get("info", {})
    prm_stats = info.get("prm_stats") or {}
    stored = prm_stats.get("prm_cost", 0.0)
    config = info.get("config", {})
    prm_name = config.get("prm_model", {}).get("model_name", "")
    pricing = MODEL_PRICING.get(prm_name)
    if pricing is None:
        return stored
    fb_log = prm_stats.get("prm_feedback_log", [])
    if any(e.get("usage") for e in fb_log):
        total = 0.0
        for e in fb_log:
            u = e.get("usage") or {}
            total += u.get("prompt_tokens", 0) * pricing["input"] + u.get("completion_tokens", 0) * pricing["output"]
        return total
    return stored


def load_traj(run_dir, iid):
    f = run_dir / iid / f"{iid}.traj.json"
    if not f.exists(): return None
    return json.loads(f.read_text())


def traj_steps(t):
    return sum(1 for m in t.get("messages", []) if m.get("role") == "assistant"
               and re.search(r'```bash\n.*?\n```', m.get("content", ""), re.DOTALL))


def classify_run(run_dir):
    trajs = list(run_dir.glob("*/*.traj.json"))
    if not trajs:
        return {}
    t = json.loads(trajs[0].read_text())
    ac = t["info"]["config"].get("agent", {})
    use_prm = ac.get("use_prm", False)
    prm_tmpl = ac.get("prm_template", "") or ""
    tmpl_hash = hashlib.md5(prm_tmpl.encode()).hexdigest()[:10] if prm_tmpl else ""
    prompt_name = PROMPT_HASH.get(tmpl_hash, tmpl_hash or "-")
    return {
        "use_prm": use_prm,
        "k": ac.get("prm_interval"),
        "prompt": prompt_name if use_prm else "-",
        "step_aware": (ac.get("step_aware_threshold") or 0) > 0,
    }


def compute_stats(run_dir, instance_ids, report_file):
    report_path = run_dir / report_file
    resolved_ids = set()
    if report_path.exists():
        resolved_ids = set(json.loads(report_path.read_text()).get("resolved_ids", []))
    resolved_in = resolved_ids & instance_ids
    exits = Counter()
    total_steps = total_cost = total_prm_cost = 0.0
    n_found = 0
    for iid in instance_ids:
        t = load_traj(run_dir, iid)
        if t is None: continue
        n_found += 1
        exits[t["info"].get("exit_status", "unknown")] += 1
        total_steps += traj_steps(t)
        total_cost += recompute_model_cost(t)
        total_prm_cost += recompute_prm_cost(t)
    submitted = exits.get("Submitted", 0)
    n = len(instance_ids)
    return {
        "resolved": len(resolved_in),
        "submitted": submitted,
        "res_rate_subset": 100 * len(resolved_in) / n if n else 0,
        "res_rate_sub": (100 * len(resolved_in) / submitted) if submitted else None,
        "avg_steps": total_steps / n if n else 0,
        "avg_model_cost": total_cost / n if n else 0,
        "avg_prm_cost": total_prm_cost / n if n else 0,
        "avg_total_cost": (total_cost + total_prm_cost) / n if n else 0,
        "n_found": n_found,
        "resolved_ids": resolved_in,
    }


def compute_fallback_stats(prm_dir, base_dir, instance_ids, report_file):
    prm_resolved = set()
    if (prm_dir / report_file).exists():
        prm_resolved = set(json.loads((prm_dir / report_file).read_text()).get("resolved_ids", []))
    base_resolved = set()
    if (base_dir / "report.json").exists():
        base_resolved = set(json.loads((base_dir / "report.json").read_text()).get("resolved_ids", []))
    fb_steps = 0
    fb_cost = 0.0
    fb_submitted = 0
    n_fallback = 0
    combined_resolved = 0
    prm_submitted = 0
    for iid in instance_ids:
        prm_t = load_traj(prm_dir, iid)
        if prm_t is None: continue
        ex = prm_t.get("info", {}).get("exit_status", "")
        if ex == "Submitted":
            prm_submitted += 1
            if iid in prm_resolved: combined_resolved += 1
        else:
            if iid in prm_resolved: combined_resolved += 1
            n_fallback += 1
            bt = load_traj(base_dir, iid)
            if bt is not None:
                if bt.get("info", {}).get("exit_status") == "Submitted":
                    fb_submitted += 1
                fb_steps += traj_steps(bt)
                fb_cost += recompute_model_cost(bt)
                if iid not in prm_resolved and iid in base_resolved:
                    combined_resolved += 1
    return {
        "combined_resolved": combined_resolved,
        "combined_submitted": prm_submitted + fb_submitted,
        "fb_steps": fb_steps,
        "fb_cost": fb_cost,
    }


def get_instance_ids(parent, want_full500=False):
    try:
        from datasets import load_dataset
        if want_full500:
            ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
        else:
            ds = load_dataset("MariusHobbhahn/swe-bench-verified-mini", split="test")
        return set(ds["instance_id"])
    except Exception:
        # Fallback: infer from reference run's directory listing
        return {p.name for p in (parent / REFERENCE_RUN).iterdir() if p.is_dir()}


# ───────────────────────────────────────────────────────────────────────

def build_rows(parent, instance_ids, n, want_full500=False):
    """Return a list of rows (each a list of strings matching HEADERS below)."""
    base_dir = parent / REFERENCE_RUN
    rows = []

    def format_stat_cells(res, sub, steps, total_cost):
        return [
            str(res), str(sub),
            f"{100 * res / n:.2f}",
            f"{100 * res / sub:.2f}" if sub else "N/A",
            f"{steps:.2f}",
            f"{total_cost:.3f}",
        ]

    # ── Baseline rows ──

    # Individual base CWM runs
    for i in range(5):
        d = parent / f"singularity_edit_obs_final_only_{i}_cwm"
        if not d.exists(): continue
        if want_full500 and not (d / "report.json").exists(): continue
        if want_full500:
            preds = d / "preds.json"
            if preds.exists():
                if len(json.loads(preds.read_text())) != 500:
                    continue
        s = compute_stats(d, instance_ids, "report.json")
        if s["n_found"] < n * 0.5: continue
        rows.append([
            "Baseline", f"Base CWM (run {i})", "-",
            *format_stat_cells(s["resolved"], s["submitted"], s["avg_steps"], s["avg_total_cost"]),
        ])

    # Base cwm-sft
    d = parent / "singularity_edit_obs_final_only_0_cwm-sft"
    if d.exists():
        s = compute_stats(d, instance_ids, "report.json")
        if s["n_found"] >= n * 0.5 and (not want_full500 or (d / "report.json").exists()):
            rows.append([
                "Baseline", "Base cwm-sft (run 0)", "-",
                *format_stat_cells(s["resolved"], s["submitted"], s["avg_steps"], s["avg_total_cost"]),
            ])

    # Best-of-5 oracle (only meaningful on mini50 — we have all 5 base runs)
    if not want_full500:
        bo5_resolved = set()
        bo5_submitted_ids = set()
        total_steps_all = total_cost_all = 0.0
        n_base = 0
        for base_dirname in BASE_RUN_DIRS:
            bd = parent / base_dirname
            if not bd.exists(): continue
            n_base += 1
            if (bd / "report.json").exists():
                bo5_resolved |= set(json.loads((bd / "report.json").read_text()).get("resolved_ids", [])) & instance_ids
            for iid in instance_ids:
                t = load_traj(bd, iid)
                if t:
                    total_steps_all += traj_steps(t)
                    total_cost_all += recompute_model_cost(t)
                    if t.get("info", {}).get("exit_status") == "Submitted":
                        bo5_submitted_ids.add(iid)
        if n_base > 0:
            rows.append([
                "Baseline", f"Best-of-{n_base} oracle", "-",
                *format_stat_cells(len(bo5_resolved), len(bo5_submitted_ids),
                                   total_steps_all / n, total_cost_all / n),
            ])

        # Base + run-1 fallback (mini50 only)
        d0 = parent / "singularity_edit_obs_final_only_0_cwm"
        d1 = parent / "singularity_edit_obs_final_only_1_cwm"
        if d0.exists() and d1.exists():
            fb = compute_fallback_stats(d0, d1, instance_ids, "report.json")
            s0 = compute_stats(d0, instance_ids, "report.json")
            # combined cost = run0 cost + run1 cost for fallback instances
            rows.append([
                "Baseline", "Base CWM + run-1 fallback", "-",
                *format_stat_cells(
                    fb["combined_resolved"], fb["combined_submitted"],
                    s0["avg_steps"] + fb["fb_steps"] / n,
                    s0["avg_total_cost"] + fb["fb_cost"] / n,
                ),
            ])

        # critic_selected
        critic_dir = parent / "critic_selected_cwm"
        if critic_dir.exists():
            cs = compute_stats(critic_dir, instance_ids, "report.json")
            if cs["n_found"] >= n * 0.5:
                # Critic cost estimate (from previous scripts)
                critic_cost = 0.0
                cr_path = critic_dir / "all_critic_results.json"
                if cr_path.exists():
                    crs = json.loads(cr_path.read_text())
                    input_tok = sum(r.get("formatted_tokens", 0) for r in crs)
                    output_tok = len(crs) * 500
                    pricing = MODEL_PRICING.get("facebook/cwm", {"input": 1.8e-7, "output": 1.8e-7})
                    critic_cost = input_tok * pricing["input"] + output_tok * pricing["output"]
                total_cost_critic = (total_cost_all / n) + (critic_cost / n if n else 0)
                rows.append([
                    "Baseline", "critic_selected (CWM critic × 5 runs)", "-",
                    *format_stat_cells(cs["resolved"], cs["submitted"],
                                       total_steps_all / n, total_cost_critic),
                ])

    # ── PRM runs ──
    def is_valid_run(d):
        """Require report.json or report-mini50.json (mini50 substitute) and enough coverage."""
        if want_full500:
            if not (d / "preds.json").exists(): return False, None
            if len(json.loads((d / "preds.json").read_text())) != 500: return False, None
            if not (d / "report.json").exists(): return False, None
            return True, "report.json"
        else:
            if (d / "report-mini50.json").exists(): return True, "report-mini50.json"
            if (d / "report.json").exists(): return True, "report.json"
            return False, None

    for d in sorted(parent.iterdir()):
        if not d.is_dir() or "_cwm_prm_" not in d.name:
            continue
        meta = classify_run(d)
        if not meta.get("use_prm"):
            continue
        ok, report_file = is_valid_run(d)
        if not ok:
            continue
        s = compute_stats(d, instance_ids, report_file)
        if s["n_found"] < n * 0.5:
            continue
        fb = compute_fallback_stats(d, base_dir, instance_ids, report_file)
        slug = d.name.split("_cwm_prm_")[-1]
        is_opus = "claude" in slug.lower()

        inf_prompt = meta["prompt"]
        k = str(meta["k"])
        sa = "-SA" if meta["step_aware"] else ""
        infer_label = f"{inf_prompt}-k{k}{sa}"

        if is_opus:
            category = "Opus Critic"
            base_model = "Opus"
            training = "-"
        else:
            category = "SFT Critic"
            base_model = SFT_NAME.get(slug, slug[:30])
            training = SFT_TRAIN_DESC.get(base_model, "?")

        # critic-only row
        rows.append([
            category, f"{base_model}, run: {infer_label}", training,
            *format_stat_cells(s["resolved"], s["submitted"], s["avg_steps"], s["avg_total_cost"]),
        ])
        # With-fallback row
        rows.append([
            category, f"{base_model}, run: {infer_label} + base fallback", training,
            *format_stat_cells(
                fb["combined_resolved"], fb["combined_submitted"],
                s["avg_steps"] + fb["fb_steps"] / n,
                s["avg_total_cost"] + fb["fb_cost"] / n,
            ),
        ])

    # Sort:
    #   Baseline → Opus Critic → SFT Critic.
    #   Inside SFT: Qwen3-8B base first, then sb-* runs, then r2e-* runs.
    def sort_key(row):
        cat = row[0]
        cat_order = {"Baseline": 0, "Opus Critic": 1, "SFT Critic": 2}[cat]
        model_cell = row[1]               # e.g. "sb-detailed-k5-flat, run: detailed-k5"
        training = row[2]
        is_fallback = 1 if "+ base fallback" in model_cell else 0
        # SFT sub-grouping
        if cat == "SFT Critic":
            base_model = model_cell.split(",")[0].strip()
            if base_model.startswith("Qwen3-8B"):
                sub_order = 0
            elif base_model.startswith("sb-"):
                sub_order = 1
            else:  # r2e-
                sub_order = 2
        else:
            sub_order = 0
        return (cat_order, sub_order, model_cell.replace(" + base fallback", ""), is_fallback)
    rows.sort(key=sort_key)
    return rows


HEADERS_TEMPLATE = [
    "Category", "Critic Model", "Training Data",
    "Resolved (/{n})", "Submitted (/{n})", "Res. Rate (/{n})", "Res. Rate (/submitted)",
    "Avg Steps (/{n})", "Avg Total Cost ($) (/{n})",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=str, default=str(RESULTS))
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    ref_dir = parent / REFERENCE_RUN
    if not ref_dir.exists():
        print(f"ERROR: reference run not found: {ref_dir}", file=sys.stderr)
        sys.exit(1)

    for label, want_full500 in [("mini50", False), ("full500", True)]:
        print(f"Building unified {label} CSV...", file=sys.stderr)
        ids = get_instance_ids(parent, want_full500=want_full500)
        n = len(ids)
        headers = [h.replace("{n}", str(n)) for h in HEADERS_TEMPLATE]
        rows = build_rows(parent, ids, n, want_full500=want_full500)
        out = parent / f"{label}_all.csv"
        with open(out, "w") as f:
            f.write(",".join(headers) + "\n")
            for r in rows:
                # Escape any accidental commas in fields with quotes
                f.write(",".join(_escape(c) for c in r) + "\n")
        print(f"  {len(rows)} rows → {out}", file=sys.stderr)


def _escape(cell: str) -> str:
    """Minimal CSV-safe quoting for commas and quotes."""
    if "," in cell or '"' in cell:
        return '"' + cell.replace('"', '""') + '"'
    return cell


if __name__ == "__main__":
    main()
