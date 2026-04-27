# PRM Intrusiveness Analysis Report

**Date**: 2026-03-30  |  **N**: 2393 samples  |  **Models**: Qwen3-8B (Base), Qwen3-8B (Base+Think), Clean-SFT, Clean-SFT-Think, RS-SFT, RS-SFT-Think

**Ground Truth**: Claude Opus PRM responses


## Quick Guide: What to Focus On

If you only have time for a few plots, look at these:

1. **`03_task_status_confusion.png`** — Shows how SFT models downgrade Critical samples to Needs Correction
2. **`05a_fpr_by_category.png`** — Shows which categories SFT has *higher* FPR than base (intrusiveness hotspots)
3. **`07c_on_track_fpr.png`** — When the agent is doing fine, how often does each model still falsely flag errors?
4. **`04_alignment.png`** — Overall severity alignment vs GT
5. **`08_guidance_length.png`** — How much longer is SFT guidance vs GT Claude Opus?


## 1. Format Compliance

*Plot: `01_format_compliance.png` | CSV: `01_format_compliance.csv`*

- **Qwen3-8B (Base)**: 68.5% fully parseable responses
- **Qwen3-8B (Base+Think)**: 67.2% fully parseable responses
- **Clean-SFT**: 96.7% fully parseable responses
- **Clean-SFT-Think**: 95.4% fully parseable responses
- **RS-SFT**: 98.7% fully parseable responses
- **RS-SFT-Think**: 96.9% fully parseable responses

Base models achieve 67–69% format compliance. SFT models range from 95–99%.


## 2. TASK_STATUS Severity Alignment

*Plots: `02_task_status_dist.png`, `03_task_status_confusion.png`, `04_alignment.png`*

| Model | Exact Match | Over-Intervention | Under-Intervention |
|-------|------------|-------------------|-------------------|
| Qwen3-8B (Base) | 35.1% | 13.4% | 22.7% |
| Qwen3-8B (Base+Think) | 35.2% | 21.0% | 13.7% |
| Clean-SFT | 76.0% | 7.0% | 15.1% |
| Clean-SFT-Think | 76.1% | 9.1% | 14.4% |
| RS-SFT | 72.5% | 10.2% | 17.2% |
| RS-SFT-Think | 70.7% | 10.5% | 17.2% |

**Key finding**: Models match GT severity 35–76% of the time. Under-intervention peaks at 23% — when the agent is critically off-track, it gets a mild nudge instead of a hard stop.

## 3. Per-Category Error Detection

*Plots: `05a_fpr_by_category.png` (FOCUS), `05b_fnr_by_category.png`, `05c_f1_by_category.png`, `05d_fpr_fnr_combined.png`*

| Category | GT+ | Qwen3-8B (Base) F1 | Qwen3-8B (Base+Think) F1 | Clean-SFT F1 | Clean-SFT-Think F1 | RS-SFT F1 | RS-SFT-Think F1 |
|----------|-----|-----|-----|-----|-----|-----|-----|
| Task Spec Viol. | 76 | 0.241 | 0.254 | 0.450 | 0.422 | 0.383 | 0.280 |
| Step Repetition | 928 | 0.608 | 0.643 | 0.833 | 0.861 | 0.826 | 0.797 |
| Termination Unaware. | 87 | 0.233 | 0.224 | 0.510 | 0.411 | 0.391 | 0.396 |
| Problem Misid. | 311 | 0.338 | 0.343 | 0.621 | 0.600 | 0.535 | 0.539 |
| Tool Selection Err. | 485 | 0.323 | 0.441 | 0.607 | 0.620 | 0.568 | 0.550 |
| Hallucinations | 77 | 0.068 | 0.151 | 0.206 | 0.204 | 0.262 | 0.219 |
| Info Processing Fail. | 1065 | 0.437 | 0.600 | 0.785 | 0.796 | 0.793 | 0.764 |
| Task Derailment | 586 | 0.321 | 0.420 | 0.599 | 0.637 | 0.536 | 0.565 |
| Goal Deviation | 127 | 0.174 | 0.128 | 0.171 | 0.288 | 0.220 | 0.211 |
| Context Handling Fail. | 392 | 0.392 | 0.419 | 0.573 | 0.582 | 0.560 | 0.548 |
| Verification Fail. | 965 | 0.721 | 0.816 | 0.908 | 0.898 | 0.905 | 0.894 |
| **Macro Average** | | 0.321 | 0.370 | 0.522 | 0.527 | 0.498 | 0.480 |

**Intrusiveness hotspots** (categories where any SFT model FPR > Base FPR by >0.02):
- **Task Spec Viol.**: worst SFT FPR=0.087 (Qwen3-8B (Base+Think)) vs Base=0.012
- **Step Repetition**: worst SFT FPR=0.088 (Qwen3-8B (Base+Think)) vs Base=0.056
- **Problem Misid.**: worst SFT FPR=0.138 (RS-SFT) vs Base=0.060
- **Tool Selection Err.**: worst SFT FPR=0.142 (Qwen3-8B (Base+Think)) vs Base=0.068
- **Hallucinations**: worst SFT FPR=0.041 (RS-SFT) vs Base=0.005
- **Info Processing Fail.**: worst SFT FPR=0.399 (RS-SFT-Think) vs Base=0.127
- **Task Derailment**: worst SFT FPR=0.149 (Qwen3-8B (Base+Think)) vs Base=0.094
- **Goal Deviation**: worst SFT FPR=0.038 (Qwen3-8B (Base+Think)) vs Base=0.013
- **Context Handling Fail.**: worst SFT FPR=0.106 (RS-SFT-Think) vs Base=0.040
- **Verification Fail.**: worst SFT FPR=0.304 (Qwen3-8B (Base+Think)) vs Base=0.128

## 4. Intervention Intensity

*Plot: `06a_flag_count_dist.png`*

| Metric | GT | Qwen3-8B (Base) | Qwen3-8B (Base+Think) | Clean-SFT | Clean-SFT-Think | RS-SFT | RS-SFT-Think |
|--------|-----|-----|-----|-----|-----|-----|-----|
| Mean flags/sample | 3.17 | 1.00 | 1.52 | 2.46 | 2.72 | 2.67 | 2.58 |


## 5. Behavior on 'On Track' Samples (N=379)

*Plot: `07c_on_track_fpr.png` (FOCUS)*

When the agent is doing fine (GT = On Track), how often does each model unnecessarily intervene?

| Model | Correctly says 'On Track' | Falsely intervenes |
|-------|--------------------------|-------------------|
| Qwen3-8B (Base) | 67.3% | 17.2% |
| Qwen3-8B (Base+Think) | 53.0% | 28.2% |
| Clean-SFT | 80.2% | 17.9% |
| Clean-SFT-Think | 72.8% | 27.2% |
| RS-SFT | 69.7% | 30.3% |
| RS-SFT-Think | 66.0% | 33.0% |

**Worst false intervention on 'On Track' samples**: RS-SFT-Think at 33% vs Qwen3-8B (Base) at 17%. This is the most damaging behavior for trajectory length — the PRM tells a correctly-working agent that it has problems when it doesn't.

## 6. Guidance Length

*Plot: `08_guidance_length.png`*

| Model | Mean Guidance Length | vs GT |
|-------|--------------------|----|
| GT (Claude Opus) | 1440 chars | -- |
| Qwen3-8B (Base) | 719 chars | -50% |
| Qwen3-8B (Base+Think) | 749 chars | -48% |
| Clean-SFT | 1682 chars | +17% |
| Clean-SFT-Think | 2017 chars | +40% |
| RS-SFT | 1786 chars | +24% |
| RS-SFT-Think | 1901 chars | +32% |

Guidance length ranges from -50% (Qwen3-8B (Base)) to +40% (Clean-SFT-Think) relative to GT.


---


## Hypothesized Causes of Longer SFT Trajectories

Based on this analysis, the SFT PRM likely causes longer agent trajectories due to a combination of these factors:

1. **Severity downgrading on Critical samples** (`03_task_status_confusion.png`): A significant fraction of Critical situations are reported as 'Needs Correction'. The agent gets a mild nudge instead of a hard stop, continuing on wrong paths longer.
2. **Info Processing false positives** (`05a_fpr_by_category.png`, `07c_on_track_fpr.png`): SFT falsely flags 'Information Processing Failures' up to 40% of the time (base: 13%). This tells the agent its reasoning is wrong when it isn't, causing unnecessary pivots.
3. **Problem Misidentification false positives** (`05a_fpr_by_category.png`): SFT FPR up to 0.14 vs Base=0.06. Telling the agent it misidentified the problem causes it to re-analyze from scratch.
4. **Verbose guidance** (`08_guidance_length.png`): SFT models generate +17% to +40% guidance text relative to Claude (base models: -50%), injecting more directive noise into the agent's context at each PRM interval.
