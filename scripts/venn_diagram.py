#!/usr/bin/env python3
"""Generate a 3-set Venn diagram: prm_issue_res vs prm_tool vs prm_tool_issue_res."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib_venn import venn3


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_singularity"

SETTINGS = {
    "prm_issue_res": "singularity_edit_obs_final_only_prm_issue_res_k5_0_cwm",
    "prm_tool": "singularity_edit_obs_final_only_prm_tool_k5_0_cwm",
    "prm_tool_issue_res": "singularity_edit_obs_final_only_prm_tool_issue_res_k5_0_cwm",
}


def load_resolved_ids(setting_dir: str) -> set:
    report_path = RESULTS_DIR / setting_dir / "report.json"
    with open(report_path) as f:
        report = json.load(f)
    return set(report["resolved_ids"])


def main():
    resolved = {name: load_resolved_ids(d) for name, d in SETTINGS.items()}

    a = resolved["prm_issue_res"]
    b = resolved["prm_tool"]
    c = resolved["prm_tool_issue_res"]

    print(f"prm_issue_res: {len(a)} resolved")
    print(f"prm_tool: {len(b)} resolved")
    print(f"prm_tool_issue_res: {len(c)} resolved")

    fig, ax = plt.subplots(figsize=(10, 8))
    v = venn3(
        [a, b, c],
        set_labels=("prm_issue_res", "prm_tool", "prm_tool_issue_res"),
        ax=ax,
    )

    # Style the labels
    for text in v.set_labels:
        if text:
            text.set_fontsize(13)
            text.set_fontweight("bold")
    for text in v.subset_labels:
        if text:
            text.set_fontsize(15)
            text.set_fontweight("bold")

    ax.set_title(
        "Resolved Instances — prm_issue_res vs prm_tool vs prm_tool_issue_res",
        fontsize=15, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    out_path = RESULTS_DIR / "venn_diagram.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
