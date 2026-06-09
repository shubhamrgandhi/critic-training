#!/usr/bin/env python3
"""Pareto plot for the full SWE-bench Verified 500.

Three points per agent (cwm, qwen3-80b, qwen32b), each WITH base-fallback:
  1. Base run 0 + run 1 fallback
  2. Best SFT critic (C/C k=10) + base 0 fallback
  3. Concise opus critic k=5 + base 0 fallback

Cost calculation (same as get_stats_full500_prefix.py):
  Model cost: recomputed from token counts via MODEL_PRICING (no cache discount).
  PRM cost:   stored `info.prm_stats.prm_cost` from the trajectory.
              For Opus this includes the cache-write premium (1.25x input),
              i.e. it's the higher, real billable cost.

Steps/cost are ADDITIVE under fallback semantics — primary's full cost is
always counted, plus fallback model cost when fallback is invoked
(primary's exit_status != "Submitted" and fallback has a traj).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from get_stats_full500_prefix import (
    recompute_model_cost, recompute_prm_cost as get_prm_cost, load_traj,
)

import os
ROOT = Path(os.environ.get(
    "RESULTS_ROOT",
    str(Path(__file__).resolve().parent.parent / "results_singularity_max_150_steps_prefix"),
))
TRAINED = "qwen3-8b-full-sft-prm-r2egym-swebench-instructions-k5-opus-distill-32k-lr5e6-multiturn"

OUT = ROOT / "plots" / "main"
OUT.mkdir(parents=True, exist_ok=True)
TOTAL = 500


def load_run(run_dir: Path) -> dict | None:
    if not run_dir.is_dir():
        return None
    preds_path = run_dir / "preds.json"
    if not preds_path.exists():
        return None
    preds = json.loads(preds_path.read_text())
    report_path = run_dir / "report.json"
    resolved_ids = set()
    if report_path.exists():
        try:
            resolved_ids = set(json.loads(report_path.read_text()).get("resolved_ids", []))
        except Exception:
            pass
    out = {}
    for inst_id in preds:
        traj = load_traj(run_dir, inst_id)
        if traj is None:
            out[inst_id] = {"has_traj": False, "submitted": False, "resolved": False,
                            "model_cost": 0.0, "prm_cost": 0.0}
            continue
        exit_status = (traj.get("info") or {}).get("exit_status", "unknown")
        out[inst_id] = {
            "has_traj": True,
            "submitted": exit_status == "Submitted",
            "resolved": inst_id in resolved_ids,
            "model_cost": recompute_model_cost(traj),
            "prm_cost": get_prm_cost(traj),
        }
    return out


def fallback_metrics(primary: dict, fb: dict) -> tuple[int, float]:
    """(resolved_count, total_cost_per_instance) using fallback semantics."""
    resolved = 0
    total_cost = 0.0
    for inst, p in primary.items():
        total_cost += p["model_cost"] + p["prm_cost"]
        if p["submitted"]:
            winner = p
        else:
            f = fb.get(inst)
            if f is not None and f["has_traj"]:
                total_cost += f["model_cost"]
                winner = f
            else:
                winner = p
        if winner["resolved"]:
            resolved += 1
    return resolved, total_cost / TOTAL


# ─── Run dirs ─────────────────────────────────────────────────────────────
def sft_dir(agent: str, k: int) -> Path:
    return ROOT / f"singularity_edit_obs_final_only_prm_issue_res_instructions_step_aware_k{k}_0_{agent}_prm_{TRAINED}"

def opus_dir(agent: str, k: int) -> Path:
    return ROOT / f"singularity_edit_obs_final_only_prm_issue_res_instructions_k{k}_0_{agent}_prm_claude-opus-4-6"


def best_variant(agent: str, primary_fn, base0: dict):
    """Pick best k variant by (resolved desc, cost asc) using fallback semantics."""
    best = None  # (resolved, cost, k, run)
    for k in (5, 10):
        run = load_run(primary_fn(agent, k))
        if run is None:
            continue
        r, c = fallback_metrics(run, base0)
        cand = (r, -c, k, run, c)
        if best is None or cand > best:
            best = cand
    if best is None:
        return None
    r, _neg_c, k, _run, c = best
    return r, c, k


def gather():
    AGENT_INFO = {
        "cwm":       {"display": "CWM",       "color": "#1f77b4"},
        "qwen3-80b": {"display": "Qwen3-80B", "color": "#ff7f0e"},
        "qwen32b":   {"display": "Qwen3-32B", "color": "#2ca02c"},
    }
    points = []  # list of (agent_display, kind, color, marker, resolved, cost)
    MARKERS = {"base": "s", "sft": "*", "opus": "o"}
    for agent, info in AGENT_INFO.items():
        base0 = load_run(ROOT / f"singularity_edit_obs_final_only_0_{agent}")
        base1 = load_run(ROOT / f"singularity_edit_obs_final_only_1_{agent}")
        if base0 is None or base1 is None:
            print(f"[skip] {agent}: missing base", file=sys.stderr)
            continue
        # base + run 1 fallback
        r, c = fallback_metrics(base0, base1)
        points.append((info["display"], "Base", info["color"], MARKERS["base"], r, c))
        # SFT critic + base 0 fallback (best of k=5, k=10 by resolved-after-fallback)
        sft_best = best_variant(agent, sft_dir, base0)
        if sft_best is not None:
            r, c, k = sft_best
            print(f"  [sft pick] {agent}: k={k}", file=sys.stderr)
            points.append((info["display"], "SFT critic (ours)", info["color"], MARKERS["sft"], r, c))
        # Opus critic + base 0 fallback (best of k=5, k=10 by resolved-after-fallback)
        opus_best = best_variant(agent, opus_dir, base0)
        if opus_best is not None:
            r, c, k = opus_best
            print(f"  [opus pick] {agent}: k={k}", file=sys.stderr)
            points.append((info["display"], "Concise Opus critic", info["color"], MARKERS["opus"], r, c))
    return points


def main():
    points = gather()
    print("\nPoints (agent, kind, resolved, $/inst):", file=sys.stderr)
    for p in points:
        print(f"  {p[0]:<10} {p[1]:<35} {p[4]:>3}/500  ${p[5]:.4f}", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Per-agent Pareto frontier: connect each agent's points (sorted by cost),
    # keeping only points that are not dominated within that agent.
    by_agent: dict[str, list] = {}
    for p in points:
        by_agent.setdefault(p[0], []).append(p)
    for agent_disp, agent_pts in by_agent.items():
        agent_pts_sorted = sorted(agent_pts, key=lambda x: x[5])
        # Keep only Pareto-optimal points (monotonically increasing resolved with cost).
        frontier = []
        best_r = -1
        for p in agent_pts_sorted:
            if p[4] > best_r:
                frontier.append(p)
                best_r = p[4]
        if len(frontier) >= 2:
            fx = [p[5] for p in frontier]
            fy = [p[4] for p in frontier]
            ax.plot(fx, fy, color=frontier[0][2], linestyle="-",
                    linewidth=1.8, alpha=0.55, zorder=1)

    for (agent_disp, kind, color, marker, r, c) in points:
        size = 720 if marker == "*" else 460
        ax.scatter([c], [r], c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=1.4, zorder=3)

    # Legend: 3 marker shapes for 3 settings, plus 3 colors for 3 agents.
    from matplotlib.lines import Line2D
    setting_legend = [
        Line2D([], [], marker="s", color="w", markerfacecolor="#888", markeredgecolor="black",
               markersize=22, label="Base"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#888", markeredgecolor="black",
               markersize=26, label="SFT critic (ours)"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#888", markeredgecolor="black",
               markersize=22, label="Concise Opus critic"),
    ]
    agent_legend = [
        Line2D([], [], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="black",
               markersize=22, label="CWM"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#ff7f0e", markeredgecolor="black",
               markersize=22, label="Qwen3-80B"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="black",
               markersize=22, label="Qwen3-32B"),
    ]
    leg1 = ax.legend(handles=setting_legend, loc="lower right", frameon=True,
                     framealpha=0.95, title="Setting", fontsize=21, title_fontsize=22)
    ax.add_artist(leg1)
    ax.legend(handles=agent_legend, loc="upper left", frameon=True,
              framealpha=0.95, fontsize=21)

    ax.set_xlabel("Avg total cost per instance ($)", fontsize=22)
    ax.set_ylabel(f"Instances resolved (out of {TOTAL})", fontsize=22)
    ax.set_title("Total cost vs. performance, SWE-bench Verified (500)", fontsize=23)
    ax.tick_params(axis="both", labelsize=19)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    max_y = max(p[4] for p in points)
    min_y = min(p[4] for p in points)
    ax.set_ylim(max(0, min_y - 25), max_y + 25)
    max_x = max(p[5] for p in points)
    ax.set_xlim(-0.02, max_x * 1.18)

    fig.tight_layout()
    fig.savefig(OUT / "pareto_full500.png", bbox_inches="tight")
    fig.savefig(OUT / "pareto_full500.pdf", bbox_inches="tight")
    print(f"\nSaved: {OUT / 'pareto_full500'}.{{png,pdf}}", file=sys.stderr)


if __name__ == "__main__":
    main()
