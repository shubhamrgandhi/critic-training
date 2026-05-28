#!/usr/bin/env python3
"""
Generate bar plots for the critic experiments.

Each bar is stacked: solid = critic-only resolved count, hatched on top =
additional gain from base fallback. The total bar height = resolved count
with fallback. Text annotation above each bar: "resolved / submitted".

Naming convention for SFT Critics (no more v1/v2):
  <source>-<teacher_prompt>-k<K>-<format>
e.g. `r2e-detailed-k5-mt`, `sb-concise-k10-flat`.
  source: sb = SWE-bench Verified distill; r2e = R2E-Gym distill
  teacher_prompt: detailed (was issue_res) or concise (was instructions)
  k: critic intervention interval during training
  format: flat (3-message flattened) or mt (real multi-turn)

Inference config is shown separately on each bar's x-label:
  "prompt-k<K>[-SA]"  (SA = step-aware wrapper at inference time)

Train vs. inference can differ — the label makes that explicit.

Output: results_singularity_max_150_steps_prefix/plots/
  headline.{png,pdf}
  opus_ablation.{png,pdf}
  sft_main.{png,pdf}
  k_flip.{png,pdf}
  sft_all_configs.{png,pdf}
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"
OUT = RESULTS / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROMPT_HASH = {"674225fc2d": "detailed", "adf350376a": "concise"}

SFT_TRAIN_NAME = {
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
    "qwen3-8b":                                                                            "Qwen3-8B-base",
}


# ─── Scan runs directly from trajectories: gives us training + inference + resolved + submitted ───

def scan_runs(parent: Path, instance_ids: set[str]):
    """Return a list of dicts, one per critic run, with all info we need for plots."""
    runs = []
    for d in sorted(parent.iterdir()):
        if not d.is_dir() or "_cwm_prm_" not in d.name:
            continue
        trajs = list(d.glob("*/*.traj.json"))
        if not trajs:
            continue
        t = json.loads(trajs[0].read_text())
        ac = t["info"]["config"].get("agent", {})
        if not ac.get("use_prm"):
            continue

        slug = d.name.split("_cwm_prm_")[-1]
        # Determine inference prompt by template hash
        tmpl = ac.get("prm_template", "") or ""
        th = hashlib.md5(tmpl.encode()).hexdigest()[:10]
        infer_prompt = PROMPT_HASH.get(th, f"other({th})")
        k_infer = ac.get("prm_interval")
        sa_infer = (ac.get("step_aware_threshold") or 0) > 0
        infer_label = f"{infer_prompt}-k{k_infer}" + ("-SA" if sa_infer else "")

        # Compute resolved/submitted on subset
        def count_for_report(report_path):
            if not report_path.exists():
                return None, None
            r = json.loads(report_path.read_text())
            resolved = len(set(r.get("resolved_ids", [])) & instance_ids)
            # Submitted: from trajectories
            submitted = 0
            for iid in instance_ids:
                traj_file = d / iid / f"{iid}.traj.json"
                if traj_file.exists():
                    tt = json.loads(traj_file.read_text())
                    if tt.get("info", {}).get("exit_status") == "Submitted":
                        submitted += 1
            return resolved, submitted

        # Prefer mini50-specific report if it exists
        if len(instance_ids) == 50 and (d / "report-mini50.json").exists():
            resolved, submitted = count_for_report(d / "report-mini50.json")
        elif (d / "report.json").exists():
            resolved, submitted = count_for_report(d / "report.json")
        else:
            continue

        if resolved is None or submitted is None:
            continue

        # Skip runs with too few instances found (incomplete)
        n_found = sum(1 for iid in instance_ids if (d / iid / f"{iid}.traj.json").exists())
        if n_found < len(instance_ids) * 0.5:
            continue

        # Compute with-fallback numbers using base run 0
        base = parent / "singularity_edit_obs_final_only_0_cwm"
        base_resolved_ids = set()
        if (base / "report.json").exists():
            base_resolved_ids = set(json.loads((base / "report.json").read_text()).get("resolved_ids", [])) & instance_ids
        combined_resolved = 0
        combined_submitted = 0
        for iid in instance_ids:
            tf = d / iid / f"{iid}.traj.json"
            if not tf.exists():
                continue
            tt = json.loads(tf.read_text())
            exit_status = tt.get("info", {}).get("exit_status", "")
            resolved_ids_run = set()
            # Need to peek at which report we used
            rp = (d / "report-mini50.json") if (len(instance_ids) == 50 and (d / "report-mini50.json").exists()) else (d / "report.json")
            if rp.exists():
                resolved_ids_run = set(json.loads(rp.read_text()).get("resolved_ids", []))
            if exit_status == "Submitted":
                combined_submitted += 1
                if iid in resolved_ids_run:
                    combined_resolved += 1
            else:
                # Fallback to base
                btf = base / iid / f"{iid}.traj.json"
                if btf.exists():
                    btt = json.loads(btf.read_text())
                    bexit = btt.get("info", {}).get("exit_status", "")
                    if bexit == "Submitted":
                        combined_submitted += 1
                    if iid in base_resolved_ids:
                        combined_resolved += 1
                # Count critic-resolved even if not submitted
                if iid in resolved_ids_run and iid not in base_resolved_ids:
                    # already counted above if in resolved_ids_run
                    pass
                if iid in resolved_ids_run and exit_status != "Submitted":
                    # add if not already counted via base_resolved
                    if iid not in base_resolved_ids:
                        combined_resolved += 1

        # Identify critic model
        is_opus = "claude" in slug.lower()
        train_name = "Opus (no SFT)" if is_opus else SFT_TRAIN_NAME.get(slug, slug[:25])

        runs.append({
            "slug": slug,
            "is_opus": is_opus,
            "train": train_name,
            "infer": infer_label,
            "infer_prompt": infer_prompt,
            "k_infer": str(k_infer),
            "sa_infer": sa_infer,
            "resolved_prm": resolved,
            "submitted_prm": submitted,
            "resolved_combined": combined_resolved,
            "submitted_combined": combined_submitted,
            "dir": d.name,
        })
    return runs


def load_instance_ids_mini50():
    from datasets import load_dataset
    return set(load_dataset("MariusHobbhahn/swe-bench-verified-mini", split="test")["instance_id"])


def load_instance_ids_full500():
    from datasets import load_dataset
    return set(load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")["instance_id"])


# ─── Baseline info ───

def get_baseline_info(parent, instance_ids):
    """Return dict with base, base+run1 fallback, best-of-5 oracle, and cwm-sft stats."""
    base_dir = parent / "singularity_edit_obs_final_only_0_cwm"
    base_r = 0
    base_s = 0
    base_resolved_ids = set()
    if (base_dir / "report.json").exists():
        base_resolved_ids = set(json.loads((base_dir / "report.json").read_text()).get("resolved_ids", [])) & instance_ids
        base_r = len(base_resolved_ids)
    for iid in instance_ids:
        tf = base_dir / iid / f"{iid}.traj.json"
        if tf.exists():
            tt = json.loads(tf.read_text())
            if tt.get("info", {}).get("exit_status") == "Submitted":
                base_s += 1

    # cwm-sft base run (different base model; no PRM)
    cwm_sft_dir = parent / "singularity_edit_obs_final_only_0_cwm-sft"
    cwm_sft_r, cwm_sft_s = None, None
    if cwm_sft_dir.exists() and (cwm_sft_dir / "report.json").exists():
        cwm_sft_r = len(set(json.loads((cwm_sft_dir / "report.json").read_text()).get("resolved_ids", [])) & instance_ids)
        cwm_sft_s = 0
        for iid in instance_ids:
            tf = cwm_sft_dir / iid / f"{iid}.traj.json"
            if tf.exists():
                tt = json.loads(tf.read_text())
                if tt.get("info", {}).get("exit_status") == "Submitted":
                    cwm_sft_s += 1

    # Base + run 1 fallback: if run 0 did not submit, try run 1's submission
    run1_dir = parent / "singularity_edit_obs_final_only_1_cwm"
    fb_r, fb_s = base_r, base_s
    if run1_dir.exists():
        run1_resolved = set()
        if (run1_dir / "report.json").exists():
            run1_resolved = set(json.loads((run1_dir / "report.json").read_text()).get("resolved_ids", [])) & instance_ids
        fb_r = 0
        fb_s = 0
        for iid in instance_ids:
            tf0 = base_dir / iid / f"{iid}.traj.json"
            if not tf0.exists():
                continue
            t0 = json.loads(tf0.read_text())
            ex0 = t0.get("info", {}).get("exit_status", "")
            if ex0 == "Submitted":
                fb_s += 1
                if iid in base_resolved_ids:
                    fb_r += 1
            else:
                # fall back to run 1
                tf1 = run1_dir / iid / f"{iid}.traj.json"
                if tf1.exists():
                    t1 = json.loads(tf1.read_text())
                    ex1 = t1.get("info", {}).get("exit_status", "")
                    if ex1 == "Submitted":
                        fb_s += 1
                    if iid in run1_resolved:
                        fb_r += 1
                elif iid in base_resolved_ids:
                    fb_r += 1

    # Best-of-5
    bo5_resolved_ids = set()
    bo5_submitted_ids = set()
    for i in range(5):
        d = parent / f"singularity_edit_obs_final_only_{i}_cwm"
        if not d.exists():
            continue
        if (d / "report.json").exists():
            bo5_resolved_ids |= set(json.loads((d / "report.json").read_text()).get("resolved_ids", [])) & instance_ids
        for iid in instance_ids:
            tf = d / iid / f"{iid}.traj.json"
            if tf.exists():
                tt = json.loads(tf.read_text())
                if tt.get("info", {}).get("exit_status") == "Submitted":
                    bo5_submitted_ids.add(iid)
    return {
        "base_r": base_r, "base_s": base_s,
        "fb_r": fb_r, "fb_s": fb_s,
        "bo5_r": len(bo5_resolved_ids), "bo5_s": len(bo5_submitted_ids),
        "cwm_sft_r": cwm_sft_r, "cwm_sft_s": cwm_sft_s,
    }


# ─── Utilities ───

def lighten(color, amount=0.55):
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def stacked_bar(ax, xs, prm_only, combined, width, color, hatch_lighten=True):
    """Draw bars: solid = prm_only, hatched = (combined - prm_only) on top."""
    light = lighten(color) if hatch_lighten else color
    for xi, p, c in zip(xs, prm_only, combined):
        if np.isnan(c) and np.isnan(p):
            continue
        if np.isnan(p):
            # No critic/fallback distinction (e.g., base or oracle)
            ax.bar(xi, c, width, color=color, edgecolor="black", linewidth=0.5)
            continue
        ax.bar(xi, p, width, color=color, edgecolor="black", linewidth=0.5)
        gain = (c if not np.isnan(c) else p) - p
        if gain > 0.01:
            ax.bar(xi, gain, width, bottom=p, color=light, edgecolor="black",
                   linewidth=0.5, hatch="//")


def annotate_tops(ax, xs, prm_only, combined, submitted_prm, submitted_combined,
                  benchmark_n, fontsize=9, dy=0.012):
    """Annotate each bar with 'resolved/submitted' for both the critic-only (midpoint
    of solid region) and the combined (above the bar)."""
    ymax_offset = benchmark_n * dy
    for xi, p, c, sp, sc in zip(xs, prm_only, combined, submitted_prm, submitted_combined):
        top = c if not np.isnan(c) else p
        if np.isnan(top):
            continue
        # Label above the bar: resolved/submitted with fallback (or base if no fallback concept)
        if sc is None or np.isnan(sc):
            label_top = f"{int(top)}"
        else:
            label_top = f"{int(top)}/{int(sc)}"
        ax.text(xi, top + ymax_offset, label_top, ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold")
        # Label on the solid portion: critic-only resolved/submitted
        if not np.isnan(p) and not np.isnan(sp) and p > benchmark_n * 0.05:
            ax.text(xi, p / 2, f"{int(p)}/{int(sp)}", ha="center", va="center",
                    fontsize=fontsize - 1, color="white", fontweight="bold")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved: {OUT / name}.{{png,pdf}}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  Pre-scan data
# ═══════════════════════════════════════════════════════════════════════

print("Scanning mini50 runs...")
mini_ids = load_instance_ids_mini50()
mini_runs = scan_runs(RESULTS, mini_ids)
mini_baseline = get_baseline_info(RESULTS, mini_ids)

print("Scanning full500 runs...")
full_ids = load_instance_ids_full500()
full_runs = scan_runs(RESULTS, full_ids)
full_baseline = get_baseline_info(RESULTS, full_ids)


# Shared description of what each config token means; printed once on each figure.
CONFIG_LEGEND = (
    "Config tokens — train: <source>-<teacher_prompt>-k<K>-<format>  |  "
    "run: <inference_prompt>-k<K>[-SA]\n"
    "  source: sb = distilled from Opus on SWE-Bench (overlaps eval) · r2e = distilled from Opus on R2E-Gym (held-out)\n"
    "  teacher_prompt / inference_prompt: detailed = long, specific, permissive  ·  concise = short, conservative, no commands\n"
    "  k = critic intervention interval  ·  format: flat = single-message trace  ·  mt = multi-turn chat  ·  SA = step-aware nudge"
)


def _add_config_legend(fig):
    fig.text(0.5, -0.02, CONFIG_LEGEND, ha="center", va="top", fontsize=8,
             family="monospace", color="#333333")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 1: Headline
# ═══════════════════════════════════════════════════════════════════════

def headline_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))

    for ax, (label, n, runs, bl) in zip(
        axes,
        [("SWE-Bench Verified Mini (50)", 50, mini_runs, mini_baseline),
         ("SWE-Bench Verified Full (500)", 500, full_runs, full_baseline)],
    ):
        opus_runs = [r for r in runs if r["is_opus"]]
        best_opus = max(opus_runs, key=lambda r: r["resolved_combined"]) if opus_runs else None
        sft_runs = [r for r in runs if not r["is_opus"] and r["train"] != "Qwen3-8B-base"]
        best_sft = max(sft_runs, key=lambda r: r["resolved_combined"]) if sft_runs else None

        bars = [("Base CWM\n(no critic)", None, bl["base_r"], None, bl["base_s"], "#888888")]
        # Best-of-5 oracle only makes sense when we have all 5 base runs (mini50)
        if n == 50:
            bars.append(("Best-of-5\noracle", None, bl["bo5_r"], None, bl["bo5_s"], "#bbbbbb"))
        if best_opus:
            bars.append((f"Opus Critic\nrun: {best_opus['infer']}",
                         best_opus["resolved_prm"], best_opus["resolved_combined"],
                         best_opus["submitted_prm"], best_opus["submitted_combined"], "#d62728"))
        if best_sft:
            bars.append((f"Qwen-8B SFT Critic\ntrain: {best_sft['train']}\nrun: {best_sft['infer']}",
                         best_sft["resolved_prm"], best_sft["resolved_combined"],
                         best_sft["submitted_prm"], best_sft["submitted_combined"], "#1f77b4"))

        xs = np.arange(len(bars))
        width = 0.6
        labels = [b[0] for b in bars]
        prm_only = [b[1] if b[1] is not None else float("nan") for b in bars]
        combined = [b[2] for b in bars]
        submitted_prm = [b[3] if b[3] is not None else float("nan") for b in bars]
        submitted_combined = [b[4] for b in bars]
        colors = [b[5] for b in bars]

        for xi, p, c, col in zip(xs, prm_only, combined, colors):
            stacked_bar(ax, [xi], [p], [c], width, col)
        annotate_tops(ax, xs, prm_only, combined, submitted_prm, submitted_combined, n)

        # Dotted reference line for base+run1 fallback (mini50 only — no run-1 full500 data)
        if n == 50 and bl["fb_r"] != bl["base_r"]:
            ax.axhline(bl["fb_r"], color="#666666", linestyle="--", linewidth=1.2)
            ax.text(len(bars) - 0.5, bl["fb_r"] + 0.5,
                    f"base + run-1 fallback: {bl['fb_r']}/{bl['fb_s']}",
                    fontsize=8, color="#666666", ha="right", va="bottom")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, n)
        ax.set_ylabel(f"Instances resolved (out of {n})")
        ax.set_title(label)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        step = 5 if n == 50 else 50
        ax.set_yticks(np.arange(0, n + 1, step))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#4d4d4d", edgecolor="black", label="Critic only"),
        Patch(facecolor=lighten("#4d4d4d"), edgecolor="black", hatch="//",
              label="Additional gain from base-run fallback"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2,
               label="Base CWM + run-1 fallback (mini50 only)"),
    ]
    fig.suptitle("Critic-guided CWM agent: resolved counts\n"
                 "(labels: bold='resolved/submitted' w/ fallback · inside bar='resolved/submitted' critic-only)",
                 y=0.99, fontsize=11)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.84])
    _add_config_legend(fig)
    save(fig, "headline")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 2: Opus ablation — training prompt × k
# ═══════════════════════════════════════════════════════════════════════

def opus_ablation_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (label, n, runs, bl) in zip(
        axes,
        [("SWE-Bench Verified Mini (50)", 50, mini_runs, mini_baseline),
         ("SWE-Bench Verified Full (500)", 500, full_runs, full_baseline)],
    ):
        opus = [r for r in runs if r["is_opus"] and not r["sa_infer"]]
        prompts = ["detailed", "concise"]
        ks = ["5", "10"]
        x = np.arange(len(prompts))
        width = 0.35

        for ki, k in enumerate(ks):
            po, co, sp, sc = [], [], [], []
            for p in prompts:
                match = next((r for r in opus if r["infer_prompt"] == p and r["k_infer"] == k), None)
                po.append(match["resolved_prm"] if match else float("nan"))
                co.append(match["resolved_combined"] if match else float("nan"))
                sp.append(match["submitted_prm"] if match else float("nan"))
                sc.append(match["submitted_combined"] if match else float("nan"))
            color = "#d62728" if k == "5" else "#ff9896"
            xs = x + (ki - 0.5) * width
            stacked_bar(ax, xs, po, co, width, color)
            annotate_tops(ax, xs, po, co, sp, sc, n)

        # Base reference lines
        ax.axhline(bl["base_r"], color="#666666", linestyle=":", linewidth=1.2,
                   label=f"Base CWM ({bl['base_r']}/{bl['base_s']})")
        if n == 50 and bl["fb_r"] != bl["base_r"]:
            ax.axhline(bl["fb_r"], color="#666666", linestyle="--", linewidth=1.2,
                       label=f"Base + run-1 fallback ({bl['fb_r']}/{bl['fb_s']})")

        ax.set_xticks(x)
        ax.set_xticklabels(prompts)
        ax.set_ylim(0, n)
        ax.set_ylabel(f"Instances resolved (out of {n})")
        ax.set_title(label)
        ax.set_xlabel("Inference prompt")
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        step = 5 if n == 50 else 50
        ax.set_yticks(np.arange(0, n + 1, step))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#d62728", edgecolor="black", label="k=5 (critic only)"),
        Patch(facecolor=lighten("#d62728"), edgecolor="black", hatch="//", label="k=5 (+ fallback)"),
        Patch(facecolor="#ff9896", edgecolor="black", label="k=10 (critic only)"),
        Patch(facecolor=lighten("#ff9896"), edgecolor="black", hatch="//", label="k=10 (+ fallback)"),
        Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.2, label="Base CWM"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2, label="Base + run-1 fallback (mini50)"),
    ]
    fig.suptitle("Opus Critic ablation: inference prompt × intervention interval",
                 y=0.98, fontsize=12)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.88])
    _add_config_legend(fig)
    save(fig, "opus_ablation")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 3: Main SFT — one bar per SFT model, best inference config
# ═══════════════════════════════════════════════════════════════════════

def sft_main_plot():
    """Best inference config per SFT model on mini50. Shows which training source works best."""
    n = 50
    bl = mini_baseline
    sft_runs_here = [r for r in mini_runs if not r["is_opus"] and r["train"] != "Qwen3-8B-base"]

    from collections import defaultdict
    by_model = defaultdict(list)
    for r in sft_runs_here:
        by_model[r["train"]].append(r)
    best_by_model = {m: max(rs, key=lambda r: r["resolved_combined"]) for m, rs in by_model.items()}

    def sort_key(m):
        return (0 if m.startswith("r2e") else 1, m)
    ordered = sorted(best_by_model.keys(), key=sort_key)
    def color_for(m):
        if m.startswith("r2esb"): return "#e6b800"  # deep yellow — R2E-Gym + SWE-Bench
        if m.startswith("r2e"):   return "#2ca02c"  # green — R2E-Gym only
        return "#1f77b4"  # blue — SWE-Bench only (overlaps eval)
    colors = [color_for(m) for m in ordered]

    labels = []
    po, co, sp, sc = [], [], [], []
    for m in ordered:
        r = best_by_model[m]
        labels.append(f"train: {m}\nrun: {r['infer']}")
        po.append(r["resolved_prm"])
        co.append(r["resolved_combined"])
        sp.append(r["submitted_prm"])
        sc.append(r["submitted_combined"])

    best_opus_mini = max(r["resolved_combined"] for r in mini_runs if r["is_opus"])

    fig, ax = plt.subplots(figsize=(max(11, 1.8 * len(labels) + 2), 6))
    x = np.arange(len(labels))
    width = 0.55
    for xi, p, c, col in zip(x, po, co, colors):
        stacked_bar(ax, [xi], [p], [c], width, col)
    annotate_tops(ax, x, po, co, sp, sc, n)

    ax.axhline(bl["base_r"], color="#666666", linestyle=":", linewidth=1.2)
    ax.axhline(bl["fb_r"], color="#666666", linestyle="--", linewidth=1.2)
    ax.axhline(best_opus_mini, color="#d62728", linestyle=":", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, n)
    ax.set_ylabel(f"Instances resolved (out of {n})")
    ax.set_title("Best-configuration Qwen-8B SFT Critics on SWE-Bench Verified Mini (50)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0, n + 1, 5))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#e6b800", edgecolor="black", label="Trained on R2E-Gym + SWE-Bench (held-out)"),
        Patch(facecolor=lighten("#e6b800"), edgecolor="black", hatch="//", label="R2E-Gym + SWE-Bench (+ fallback)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Trained on R2E-Gym only (held-out)"),
        Patch(facecolor=lighten("#2ca02c"), edgecolor="black", hatch="//", label="R2E-Gym only (+ fallback)"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Trained on SWE-Bench (overlaps eval)"),
        Patch(facecolor=lighten("#1f77b4"), edgecolor="black", hatch="//", label="SWE-Bench (+ fallback)"),
        Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.2,
               label=f"Base CWM ({bl['base_r']}/{bl['base_s']})"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2,
               label=f"Base + run-1 fallback ({bl['fb_r']}/{bl['fb_s']})"),
        Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.2,
               label=f"Best Opus Critic ({best_opus_mini}/50)"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    _add_config_legend(fig)
    save(fig, "sft_main")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 4: k-flip
# ═══════════════════════════════════════════════════════════════════════

def k_flip_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (label, n, runs, bl) in zip(
        axes,
        [("SWE-Bench Verified Mini (50)", 50, mini_runs, mini_baseline),
         ("SWE-Bench Verified Full (500)", 500, full_runs, full_baseline)],
    ):
        opus_k5 = next((r for r in runs if r["is_opus"] and r["infer_prompt"] == "detailed"
                        and r["k_infer"] == "5" and not r["sa_infer"]), None)
        opus_k10 = next((r for r in runs if r["is_opus"] and r["infer_prompt"] == "detailed"
                         and r["k_infer"] == "10" and not r["sa_infer"]), None)
        sft_k5 = next((r for r in runs if r["train"] == "sb-detailed-k5-flat"
                       and r["infer_prompt"] == "detailed" and r["k_infer"] == "5"
                       and not r["sa_infer"]), None)
        sft_k10 = next((r for r in runs if r["train"] == "sb-detailed-k5-flat"
                        and r["infer_prompt"] == "detailed" and r["k_infer"] == "10"
                        and not r["sa_infer"]), None)

        ks = ["5", "10"]
        x = np.arange(len(ks))
        width = 0.35

        def pull(match, key):
            return match[key] if match else float("nan")

        opus_po = [pull(opus_k5, "resolved_prm"), pull(opus_k10, "resolved_prm")]
        opus_co = [pull(opus_k5, "resolved_combined"), pull(opus_k10, "resolved_combined")]
        opus_sp = [pull(opus_k5, "submitted_prm"), pull(opus_k10, "submitted_prm")]
        opus_sc = [pull(opus_k5, "submitted_combined"), pull(opus_k10, "submitted_combined")]
        sft_po = [pull(sft_k5, "resolved_prm"), pull(sft_k10, "resolved_prm")]
        sft_co = [pull(sft_k5, "resolved_combined"), pull(sft_k10, "resolved_combined")]
        sft_sp = [pull(sft_k5, "submitted_prm"), pull(sft_k10, "submitted_prm")]
        sft_sc = [pull(sft_k5, "submitted_combined"), pull(sft_k10, "submitted_combined")]

        xs_opus = x - width/2
        xs_sft = x + width/2
        stacked_bar(ax, xs_opus, opus_po, opus_co, width, "#d62728")
        stacked_bar(ax, xs_sft, sft_po, sft_co, width, "#1f77b4")
        annotate_tops(ax, xs_opus, opus_po, opus_co, opus_sp, opus_sc, n)
        annotate_tops(ax, xs_sft, sft_po, sft_co, sft_sp, sft_sc, n)

        # Base + fallback reference lines
        ax.axhline(bl["base_r"], color="#666666", linestyle=":", linewidth=1.2)
        if n == 50 and bl["fb_r"] != bl["base_r"]:
            ax.axhline(bl["fb_r"], color="#666666", linestyle="--", linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_xlabel("Inference intervention interval")
        ax.set_ylim(0, n)
        ax.set_ylabel(f"Instances resolved (out of {n})")
        ax.set_title(label)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        step = 5 if n == 50 else 50
        ax.set_yticks(np.arange(0, n + 1, step))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#d62728", edgecolor="black", label="Opus Critic (critic only)"),
        Patch(facecolor=lighten("#d62728"), edgecolor="black", hatch="//", label="Opus Critic (+ fallback)"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="SFT Critic sb-detailed-k5-flat (critic only)"),
        Patch(facecolor=lighten("#1f77b4"), edgecolor="black", hatch="//", label="SFT Critic (+ fallback)"),
        Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.2, label="Base CWM"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2, label="Base + run-1 fallback (mini50)"),
    ]
    fig.suptitle("Intervention frequency: good critic prefers k=5, weaker critic prefers k=10",
                 y=0.98, fontsize=12)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.88])
    _add_config_legend(fig)
    save(fig, "k_flip")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 5: All SFT configs (supplementary)
# ═══════════════════════════════════════════════════════════════════════

def sft_all_configs_plot():
    """Every (SFT model × inference config) combination, for appendix/deep-dive."""
    n = 50
    bl = mini_baseline
    sft_runs_here = [r for r in mini_runs if not r["is_opus"] and r["train"] != "Qwen3-8B-base"]

    def sort_key(r):
        src_order = 0 if r["train"].startswith("r2e") else 1
        return (src_order, r["train"], r["infer_prompt"], int(r["k_infer"]), r["sa_infer"])
    sft_runs_here.sort(key=sort_key)

    labels = [f"train: {r['train']}\nrun:    {r['infer']}" for r in sft_runs_here]
    po = [r["resolved_prm"] for r in sft_runs_here]
    co = [r["resolved_combined"] for r in sft_runs_here]
    sp = [r["submitted_prm"] for r in sft_runs_here]
    sc = [r["submitted_combined"] for r in sft_runs_here]
    def color_for_train(t):
        if t.startswith("r2esb"): return "#e6b800"
        if t.startswith("r2e"):   return "#2ca02c"
        return "#1f77b4"
    colors = [color_for_train(r["train"]) for r in sft_runs_here]

    best_opus_combined = max(r["resolved_combined"] for r in mini_runs if r["is_opus"])

    fig, ax = plt.subplots(figsize=(max(14, 0.8 * len(labels) + 2), 7))
    x = np.arange(len(labels))
    width = 0.65
    for xi, p, c, col in zip(x, po, co, colors):
        stacked_bar(ax, [xi], [p], [c], width, col)
    annotate_tops(ax, x, po, co, sp, sc, n)

    ax.axhline(bl["base_r"], color="#666666", linestyle=":", linewidth=1.2)
    ax.axhline(bl["fb_r"], color="#666666", linestyle="--", linewidth=1.2)
    ax.axhline(best_opus_combined, color="#d62728", linestyle=":", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, n)
    ax.set_ylabel(f"Instances resolved (out of {n})")
    ax.set_title("All Qwen-8B SFT Critic configurations on SWE-Bench Verified Mini (50)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0, n + 1, 5))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#e6b800", edgecolor="black", label="Trained on R2E-Gym + SWE-Bench (held-out)"),
        Patch(facecolor=lighten("#e6b800"), edgecolor="black", hatch="//", label="R2E-Gym + SWE-Bench (+ fallback)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Trained on R2E-Gym only (held-out from eval)"),
        Patch(facecolor=lighten("#2ca02c"), edgecolor="black", hatch="//", label="R2E-Gym only (+ fallback)"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Trained on SWE-Bench (overlaps eval)"),
        Patch(facecolor=lighten("#1f77b4"), edgecolor="black", hatch="//", label="SWE-Bench (+ fallback)"),
        Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.2,
               label=f"Base CWM ({bl['base_r']}/{bl['base_s']})"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2,
               label=f"Base + run-1 fallback ({bl['fb_r']}/{bl['fb_s']})"),
        Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.2,
               label=f"Best Opus Critic ({best_opus_combined}/50)"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    _add_config_legend(fig)
    save(fig, "sft_all_configs")


# ═══════════════════════════════════════════════════════════════════════
#  Plot 6: Base-agent comparison — CWM vs cwm-sft, plus the headline stuff
#  Shows that swapping the base agent does not help; critics are the lever.
# ═══════════════════════════════════════════════════════════════════════

def base_comparison_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))

    for ax, (label, n, runs, bl) in zip(
        axes,
        [("SWE-Bench Verified Mini (50)", 50, mini_runs, mini_baseline),
         ("SWE-Bench Verified Full (500)", 500, full_runs, full_baseline)],
    ):
        opus_runs = [r for r in runs if r["is_opus"]]
        best_opus = max(opus_runs, key=lambda r: r["resolved_combined"]) if opus_runs else None
        sft_runs = [r for r in runs if not r["is_opus"] and r["train"] != "Qwen3-8B-base"]
        best_sft = max(sft_runs, key=lambda r: r["resolved_combined"]) if sft_runs else None

        bars = [("Base CWM\n(no critic)", None, bl["base_r"], None, bl["base_s"], "#888888")]
        if bl.get("cwm_sft_r") is not None:
            bars.append(("Base cwm-sft\n(no critic)", None, bl["cwm_sft_r"], None, bl["cwm_sft_s"], "#a9a9a9"))
        if n == 50:
            bars.append(("Best-of-5\noracle", None, bl["bo5_r"], None, bl["bo5_s"], "#bbbbbb"))
        if best_opus:
            bars.append((f"Opus Critic\nrun: {best_opus['infer']}",
                         best_opus["resolved_prm"], best_opus["resolved_combined"],
                         best_opus["submitted_prm"], best_opus["submitted_combined"], "#d62728"))
        if best_sft:
            bars.append((f"Qwen-8B SFT Critic\ntrain: {best_sft['train']}\nrun: {best_sft['infer']}",
                         best_sft["resolved_prm"], best_sft["resolved_combined"],
                         best_sft["submitted_prm"], best_sft["submitted_combined"], "#1f77b4"))

        xs = np.arange(len(bars))
        width = 0.6
        labels = [b[0] for b in bars]
        prm_only = [b[1] if b[1] is not None else float("nan") for b in bars]
        combined = [b[2] for b in bars]
        submitted_prm = [b[3] if b[3] is not None else float("nan") for b in bars]
        submitted_combined = [b[4] for b in bars]
        colors = [b[5] for b in bars]

        for xi, p, c, col in zip(xs, prm_only, combined, colors):
            stacked_bar(ax, [xi], [p], [c], width, col)
        annotate_tops(ax, xs, prm_only, combined, submitted_prm, submitted_combined, n)

        if n == 50 and bl["fb_r"] != bl["base_r"]:
            ax.axhline(bl["fb_r"], color="#666666", linestyle="--", linewidth=1.2)
            ax.text(len(bars) - 0.5, bl["fb_r"] + 0.5,
                    f"base + run-1 fallback: {bl['fb_r']}/{bl['fb_s']}",
                    fontsize=8, color="#666666", ha="right", va="bottom")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, n)
        ax.set_ylabel(f"Instances resolved (out of {n})")
        ax.set_title(label)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        step = 5 if n == 50 else 50
        ax.set_yticks(np.arange(0, n + 1, step))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#4d4d4d", edgecolor="black", label="Critic only"),
        Patch(facecolor=lighten("#4d4d4d"), edgecolor="black", hatch="//",
              label="Additional gain from base-run fallback"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2,
               label="Base CWM + run-1 fallback (mini50 only)"),
    ]
    fig.suptitle("Base-agent comparison: CWM vs cwm-sft, plus best critics\n"
                 "(swapping the base agent alone does not help; critics are where gains come from)",
                 y=0.99, fontsize=11)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.84])
    _add_config_legend(fig)
    save(fig, "base_comparison")


# ═══════════════════════════════════════════════════════════════════════
#  Generate all plots
# ═══════════════════════════════════════════════════════════════════════

headline_plot()
opus_ablation_plot()
sft_main_plot()
k_flip_plot()
sft_all_configs_plot()
base_comparison_plot()


# ═══════════════════════════════════════════════════════════════════════
#  Plot 7: Pareto — resolved (w/ fallback) vs cost. With-fallback points only.
# ═══════════════════════════════════════════════════════════════════════

def pareto_plot():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))

    for ax, (label, n, runs, bl) in zip(
        axes,
        [("SWE-Bench Verified Mini (50)", 50, mini_runs, mini_baseline),
         ("SWE-Bench Verified Full (500)", 500, full_runs, full_baseline)],
    ):
        # Assemble points: (name, resolved_w_fb, total_cost_w_fb, category)
        # For critic runs: total_cost_w_fb = avg_total_cost + fb_cost/n (already computed in scan_runs? let's use CSV-style compute)
        # scan_runs doesn't carry cost — compute on the fly from config + reports.
        # Simpler: load the unified CSV we already write.
        import csv
        csv_path = RESULTS / ("mini50_all.csv" if n == 50 else "full500_all.csv")
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(open(csv_path)))
        res_key = f"Resolved (/{n})"
        cost_key = f"Avg Total Cost ($) (/{n})"

        def parse_row(row):
            try:
                return int(row[res_key]), float(row[cost_key])
            except (ValueError, KeyError):
                return None

        # Per user filtering for pareto plot:
        #   - drop flat-format SFTs (user: "ignore all flattened settings")
        #   - drop rejection-sampled (RS) variants
        #   - drop v1 SFT variants (sb-detailed-k5 series; already covered by "-flat" drop)
        #   - drop step-aware-explicit / postprocess / dedup variants
        #   - Opus: keep only concise-k5
        #   - Qwen3-8B base: KEEP (shows untrained critic hurts)
        def is_exclude_critic(short):
            base_model = short.split(",")[0].strip()
            infer = short.split(", run:", 1)[1].strip() if ", run:" in short else ""
            if "-flat" in base_model:
                return True
            if "RS" in base_model:
                return True
            # Drop inference-config-specific special runs (not in current data but defensive)
            if "step_aware_explicit" in infer or "postprocess" in infer or "dedup" in infer:
                return True
            if base_model == "Opus":
                if infer != "concise-k5":
                    return True
            return False

        candidates = []
        for row in rows:
            cat = row["Category"]
            model_cell = row["Critic Model"]
            if cat == "Baseline":
                continue
            if "+ base fallback" not in model_cell:
                continue
            p = parse_row(row)
            if p is None:
                continue
            resolved, cost = p
            short = model_cell.replace(" + base fallback", "")
            if is_exclude_critic(short):
                continue
            base_model = short.split(",")[0].strip()
            if cat == "Opus Critic":
                group = "opus"
            elif base_model.startswith("Qwen3-8B"):
                group = "qwen_base"
            elif base_model.startswith("r2esb"):
                group = "sft_r2esb"
            elif base_model.startswith("r2e"):
                group = "sft_r2e"
            elif base_model.startswith("sb"):
                group = "sft_sb"
            else:
                group = "sft_other"
            candidates.append({"label": short, "resolved": resolved,
                               "cost": max(cost, 0.01), "group": group})

        # For each group, keep the best (highest resolved) — except for r2esb,
        # where we want to show ALL variants since this is the headline contribution.
        by_group_best = {}
        r2esb_all = []
        for c in candidates:
            if c["group"] == "sft_r2esb":
                r2esb_all.append(c)
            else:
                cur = by_group_best.get(c["group"])
                if cur is None or c["resolved"] > cur["resolved"]:
                    by_group_best[c["group"]] = c

        points = []
        # Baseline anchor first
        base_anchor = next((r for r in rows if r["Critic Model"] == "Base CWM + run-1 fallback"), None)
        if base_anchor is None:
            base_anchor = next((r for r in rows if r["Critic Model"] == "Base CWM (run 0)"), None)
        if base_anchor and parse_row(base_anchor):
            resolved, cost = parse_row(base_anchor)
            points.append({"label": base_anchor["Critic Model"], "resolved": resolved,
                           "cost": max(cost, 0.01), "group": "baseline"})
        # Best of each non-r2esb critic group (Qwen base → sb → r2e → Opus)
        for group in ("qwen_base", "sft_sb", "sft_r2e", "sft_other", "opus"):
            if group in by_group_best:
                points.append(by_group_best[group])
        # All r2esb variants (sorted by cost for stable label staggering)
        for c in sorted(r2esb_all, key=lambda x: x["cost"]):
            points.append(c)

        GROUP_STYLE = {
            "baseline":   {"color": "#666666", "marker": "s", "size": 90,  "label_legend": "Baseline (w/ fallback)"},
            "qwen_base":  {"color": "#9467bd", "marker": "v", "size": 80,  "label_legend": "Qwen-8B base (no SFT)"},
            "opus":       {"color": "#d62728", "marker": "o", "size": 90,  "label_legend": "Opus Critic"},
            "sft_sb":     {"color": "#1f77b4", "marker": "^", "size": 80,  "label_legend": "SFT sb-* (overlaps eval)"},
            "sft_r2e":    {"color": "#2ca02c", "marker": "D", "size": 80,  "label_legend": "SFT r2e-* (held-out)"},
            "sft_r2esb":  {"color": "#e6b800", "marker": "P", "size": 100, "label_legend": "SFT r2esb-* (R2E-Gym + SWE-Bench, held-out)"},
            "sft_other":  {"color": "#999999", "marker": "x", "size": 60,  "label_legend": "SFT other"},
        }

        # Compute Pareto frontier (maximize resolved, minimize cost)
        # A point is dominated if another point has higher resolved AND lower cost (non-strict).
        def on_pareto(pt, pts):
            for other in pts:
                if other is pt: continue
                if other["resolved"] >= pt["resolved"] and other["cost"] <= pt["cost"]:
                    if other["resolved"] > pt["resolved"] or other["cost"] < pt["cost"]:
                        return False
            return True

        pareto_pts = [pt for pt in points if on_pareto(pt, points)]
        pareto_pts.sort(key=lambda pt: pt["cost"])

        # Frontier line
        if pareto_pts:
            pareto_x = [pt["cost"] for pt in pareto_pts]
            pareto_y = [pt["resolved"] for pt in pareto_pts]
            ax.plot(pareto_x, pareto_y, color="#444444", linestyle="-",
                    linewidth=1.2, alpha=0.5, zorder=1, label="Pareto frontier")

        # Scatter all points, grouped for legend
        by_group = {}
        for pt in points:
            by_group.setdefault(pt["group"], []).append(pt)

        for group, g_pts in by_group.items():
            style = GROUP_STYLE[group]
            xs = [pt["cost"] for pt in g_pts]
            ys = [pt["resolved"] for pt in g_pts]
            ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                       s=style["size"], edgecolors="black", linewidths=0.6,
                       alpha=0.85, zorder=3, label=style["label_legend"])

        # Annotate every point. Put every label ABOVE the frontier line at large
        # vertical offsets, staggered so that labels at similar costs don't collide.
        # Order points by cost; assign staggered dy values: 40, 80, 40, 80, ...
        # This guarantees adjacent labels live at different heights above the points.
        # Labels going above match the upward slope of the frontier, so no label box
        # crosses the line.
        panel_title = label
        sorted_pts = sorted(points, key=lambda pt: pt["cost"])
        stagger = [40, 78, 40, 78, 40, 78]  # alternating heights in screen points
        for idx, pt in enumerate(sorted_pts):
            lbl = pt["label"]
            if ", run:" in lbl:
                model, infer = lbl.split(", run:", 1)
                lbl = f"{model.strip()}\n(run: {infer.strip()})"
            dy = stagger[idx % len(stagger)]
            ax.annotate(lbl, (pt["cost"], pt["resolved"]),
                        xytext=(0, dy), textcoords="offset points",
                        fontsize=9, ha="center", va="bottom",
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#888888", lw=0.7, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))

        # Linear x-scale — natural spacing; no log crowding
        ax.set_xlabel("Avg total cost per instance ($)")
        ax.set_ylabel(f"Instances resolved with fallback (out of {n})")
        ax.set_title(panel_title)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

        # y-lim: extra headroom for the stacked labels above points
        max_y = max(pt["resolved"] for pt in points) if points else n
        ax.set_ylim(0, min(n, max_y + (n * 0.35)))
        # x-lim: small left margin, 15% right margin
        max_x = max(pt["cost"] for pt in points) if points else 1.0
        ax.set_xlim(-max_x * 0.03, max_x * 1.15)

        ax.legend(loc="lower right", frameon=True, fontsize=8.5, framealpha=0.9)

    fig.suptitle("Pareto: resolved (with fallback) vs. avg total cost per instance",
                 y=0.99, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.95])
    _add_config_legend(fig)
    save(fig, "pareto")


def pareto_plot_minimal():
    """Minimal Pareto plot for SWE-Bench Verified Mini (50), 4 hand-picked points.

    Points (all with base fallback):
      1. CWM base                       = Base CWM + run-1 fallback
      2. CWM + Qwen critic (untrained)  = Qwen3-8B-base as critic, w/ fallback
      3. CWM + SFT Qwen critic          = r2egym-swebench-k5 multiturn SFT
                                          (trained on R2E-Gym + SWE-Smith, held-out)
      4. CWM + Opus critic (instructions k=5)
    """
    import csv
    csv_path = RESULTS / "mini50_all.csv"
    rows = list(csv.DictReader(open(csv_path)))
    n = 50
    res_key, cost_key = f"Resolved (/{n})", f"Avg Total Cost ($) (/{n})"

    def find(predicate):
        for r in rows:
            if predicate(r):
                try:
                    return (r["Critic Model"], int(r[res_key]), float(r[cost_key]))
                except (KeyError, ValueError):
                    return None
        return None

    # The four exact points
    pt_base    = find(lambda r: r["Critic Model"] == "Base CWM + run-1 fallback")
    pt_untrain = find(lambda r: "Qwen3-8B (base)" in r["Critic Model"] and "+ base fallback" in r["Critic Model"])
    pt_sft     = find(lambda r: "r2esb-detailed-k5-mt" in r["Critic Model"] and "+ base fallback" in r["Critic Model"])
    pt_opus    = find(lambda r: r["Category"] == "Opus Critic"
                      and "+ base fallback" in r["Critic Model"]
                      and "concise-k5" in r["Critic Model"])

    # Friendly labels (as user requested). Linear x-scale means the 4 points spread
    # out naturally, so simple above/below offsets are enough to avoid overlap.
    #   CWM base       (x=0.164, y=14): label above
    #   Qwen untrained (x=0.39,  y=10): label below (point is below frontier anyway)
    #   SFT Qwen       (x=0.642, y=19): label above
    #   Opus           (x=1.422, y=22): label above
    # Each entry: (label, resolved, cost, color, marker, dx, dy, ha, va)
    pts = []
    if pt_base:
        _, r, c = pt_base
        pts.append(("CWM base", r, max(c, 0.01), "#666666", "s",
                    0, 24, "center", "bottom"))
    if pt_untrain:
        _, r, c = pt_untrain
        pts.append(("CWM + Qwen3-8B critic\n(untrained)", r, max(c, 0.01),
                    "#9467bd", "v", 0, -24, "center", "top"))
    if pt_sft:
        _, r, c = pt_sft
        pts.append(("CWM + SFT Qwen3-8B critic\n(ours)", r, max(c, 0.01),
                    "#2ca02c", "D", 0, 28, "center", "bottom"))
    if pt_opus:
        _, r, c = pt_opus
        pts.append(("CWM + Opus critic\n(teacher)", r, max(c, 0.01),
                    "#d62728", "o", 0, 28, "center", "bottom"))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Pareto frontier line (sorted by cost)
    pts_sorted = sorted(pts, key=lambda p: p[2])
    frontier, best_r = [], -1
    for p in pts_sorted:
        if p[1] > best_r:
            frontier.append(p)
            best_r = p[1]
    fx = [p[2] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, color="#444444", linestyle="-", linewidth=1.5,
            alpha=0.55, zorder=1, label="Pareto frontier")

    # Scatter + annotate each point using per-point offsets
    for (lbl, r, c, col, mk, dx, dy, ha, va) in pts:
        ax.scatter([c], [r], c=col, marker=mk, s=240,
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.annotate(f"{lbl}\n({r}/50 resolved)",
                    (c, r), xytext=(dx, dy), textcoords="offset points",
                    fontsize=11, ha=ha, va=va, color="black",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec=col, lw=1.3, alpha=0.97),
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.2))

    ax.set_xlabel("Avg total cost per instance ($)", fontsize=11)
    ax.set_ylabel("Instances resolved (out of 50)", fontsize=11)
    ax.set_title("Critic cost vs. performance — SWE-Bench Verified Mini", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    max_y = max(p[1] for p in pts)
    ax.set_ylim(-2, max_y + 12)
    max_x = max(p[2] for p in pts)
    ax.set_xlim(0, max_x * 1.18)

    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "pareto_minimal")


def pareto_plot_minimal_critic_cost():
    """Same 4-point minimal plot, but x-axis is CRITIC-ONLY cost per instance
    (not the full end-to-end cost). Tells the story 'what do we pay for the critic itself'.

    Critic cost is recomputed from each run's trajectories using MODEL_PRICING;
    for Opus, stored `prm_cost` (from the trajectory JSON) is used.
    """
    n = 50
    try:
        from datasets import load_dataset
        mini = set(load_dataset("MariusHobbhahn/swe-bench-verified-mini", split="test")["instance_id"])
    except Exception:
        base_ref = RESULTS / "singularity_edit_obs_final_only_0_cwm"
        mini = {p.name for p in base_ref.iterdir() if p.is_dir()}

    # Re-use MODEL_PRICING dict from the SFT mapping — fall back to stored cost when unknown.
    PRICING = {
        "Qwen/Qwen3-8B": {"input": 5e-8, "output": 1.5e-7},
    }
    for slug in SFT_TRAIN_NAME.keys():
        PRICING[f"shubhamrgandhi/{slug}"] = {"input": 5e-8, "output": 1.5e-7}

    def critic_cost_for_run(run_dir: Path):
        """Avg critic cost per instance on the mini50 subset."""
        if not run_dir.exists():
            return None
        total = 0.0
        n_found = 0
        for iid in mini:
            tf = run_dir / iid / f"{iid}.traj.json"
            if not tf.exists(): continue
            try:
                t = json.loads(tf.read_text())
            except Exception:
                continue
            n_found += 1
            ps = t.get("info", {}).get("prm_stats") or {}
            stored = ps.get("prm_cost", 0.0)
            # Try to recompute from usage if pricing is known
            cfg = t.get("info", {}).get("config", {})
            prm_name = cfg.get("prm_model", {}).get("model_name", "") or ""
            pricing = PRICING.get(prm_name)
            if pricing and any(e.get("usage") for e in ps.get("prm_feedback_log", [])):
                s = 0.0
                for e in ps.get("prm_feedback_log", []):
                    u = e.get("usage") or {}
                    s += u.get("prompt_tokens", 0) * pricing["input"] + \
                         u.get("completion_tokens", 0) * pricing["output"]
                total += s
            else:
                total += stored
        return total / len(mini) if mini else None

    # The four runs (same as pareto_minimal)
    runs = {
        "base": None,  # no critic: 0 cost
        "untrain": RESULTS / "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm_prm_qwen3-8b",
        "sft":    RESULTS / "singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k10_0_cwm_prm_qwen3-8b-full-sft-prm-r2egym-swebench-k5-opus-distill-32k-lr5e6-multiturn",
        "opus":   RESULTS / "singularity_edit_obs_final_only_prm_issue_res_instructions_k5_0_cwm_prm_claude-opus-4-6",
    }

    # Resolved counts (with-fallback) — reuse the unified CSV
    import csv
    rows = list(csv.DictReader(open(RESULTS / "mini50_all.csv")))
    def find_r(predicate):
        for r in rows:
            if predicate(r):
                try:
                    return int(r[f"Resolved (/{n})"])
                except (KeyError, ValueError):
                    return None
        return None
    r_base    = find_r(lambda r: r["Critic Model"] == "Base CWM + run-1 fallback")
    r_untrain = find_r(lambda r: "Qwen3-8B (base)" in r["Critic Model"] and "+ base fallback" in r["Critic Model"])
    r_sft     = find_r(lambda r: "r2esb-detailed-k5-mt" in r["Critic Model"] and "+ base fallback" in r["Critic Model"])
    r_opus    = find_r(lambda r: r["Category"] == "Opus Critic"
                       and "+ base fallback" in r["Critic Model"]
                       and "concise-k5" in r["Critic Model"])

    c_base    = 0.0
    c_untrain = critic_cost_for_run(runs["untrain"])
    c_sft     = critic_cost_for_run(runs["sft"])
    c_opus    = critic_cost_for_run(runs["opus"])

    # Critic-only costs span ~$0 → $1.10 (Opus is 50–100x the SFT critic cost).
    # Use a log x-axis with a small floor for the zero-cost baseline so it still
    # renders. Stagger label heights so closely-costed points don't collide.
    # For log scale, nudge the 0-cost baseline to a small positive value so it's plottable.
    BASE_PLOT_X = 0.003

    pts = []
    if r_base is not None:
        pts.append(("CWM base\n(no critic)", r_base, BASE_PLOT_X, "#666666", "s"))
    if r_untrain is not None and c_untrain is not None:
        pts.append(("CWM + Qwen3-8B critic\n(untrained)", r_untrain, max(c_untrain, BASE_PLOT_X),
                    "#9467bd", "v"))
    if r_sft is not None and c_sft is not None:
        pts.append(("CWM + SFT Qwen3-8B critic\n(ours)", r_sft, max(c_sft, BASE_PLOT_X),
                    "#2ca02c", "D"))
    if r_opus is not None and c_opus is not None:
        pts.append(("CWM + Opus critic\n(teacher)", r_opus, max(c_opus, BASE_PLOT_X),
                    "#d62728", "o"))

    fig, ax = plt.subplots(figsize=(11, 7))

    # Pareto frontier
    pts_sorted = sorted(pts, key=lambda p: p[2])
    frontier, best_r = [], -1
    for p in pts_sorted:
        if p[1] > best_r:
            frontier.append(p)
            best_r = p[1]
    fx = [p[2] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, color="#444444", linestyle="-", linewidth=1.5,
            alpha=0.55, zorder=1, label="Pareto frontier")

    # Staggered vertical offsets (screen points) so close-cost labels don't collide
    stagger = [42, 90, 42, 90]
    for idx, (lbl, r, c, col, mk) in enumerate(pts_sorted):
        ax.scatter([c], [r], c=col, marker=mk, s=240,
                   edgecolors="black", linewidths=1.2, zorder=3)
        dy = stagger[idx % len(stagger)]
        ax.annotate(f"{lbl}\n({r}/50 resolved)",
                    (c, r), xytext=(0, dy), textcoords="offset points",
                    fontsize=11, ha="center", va="bottom", color="black",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec=col, lw=1.3, alpha=0.97),
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.2))

    ax.set_xscale("log")
    ax.set_xlabel("Avg critic-only cost per instance ($, log scale)", fontsize=11)
    ax.set_ylabel("Instances resolved (out of 50)", fontsize=11)
    ax.set_title("Critic-only cost vs. performance — SWE-Bench Verified Mini", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    max_y = max(p[1] for p in pts)
    ax.set_ylim(0, max_y + 22)
    # Log-scale x-lims: give some padding on both sides
    xs = [p[2] for p in pts]
    ax.set_xlim(min(xs) * 0.4, max(xs) * 2.5)

    # Note at the bottom that CWM base has zero critic cost but is plotted at a
    # small positive value so it's visible on the log axis.
    ax.text(BASE_PLOT_X, -1.5, "(no critic, cost = $0)",
            fontsize=8, color="#666666", ha="center", va="top")

    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "pareto_minimal_critic_cost")


pareto_plot()
pareto_plot_minimal()
pareto_plot_minimal_critic_cost()

print(f"\nDone. Plots in: {OUT}")
