# PRM Model Ranking

**N = 2393 samples | Ground Truth = Claude Opus 4.6 PRM responses**

---

## Metric Definitions

| Metric | What it measures | Better = |
|--------|-----------------|----------|
| **Format** | % of responses where all 12 error categories were parseable AND TASK_STATUS was present. If this is low, the model is outputting garbage instead of a structured PRM evaluation. | Higher |
| **Align** | % of responses where the model's TASK_STATUS exactly matches Claude's (On Track / Needs Correction / Critical Intervention). Measures how well the model agrees with the ground truth severity judgment. | Higher |
| **Over-Int** | % of responses where the model escalates severity above Claude's — e.g. says "Critical" when Claude said "Needs Correction", or flags errors when Claude said "On Track". The agent gets unnecessarily interrupted. | Lower |
| **Under-Int** | % of responses where the model de-escalates below Claude's — e.g. says "Needs Correction" when Claude said "Critical". The agent gets a mild nudge instead of a hard stop. | Lower |
| **On-Track FPI** | False positive intervention rate on samples where Claude said "On Track". When the agent is working correctly, how often does the model still tell it something is wrong? Directly causes unnecessary agent pivots and extra steps. | Lower |
| **Macro F1** | Average F1 score across all 12 error categories (e.g. Step Repetition, Problem Misidentification, Info Processing Failures, etc.). Measures how precisely and completely the model detects the right error types. | Higher |
| **Guidance vs GT** | How much longer or shorter the model's OVERALL_GUIDANCE text is compared to Claude's. More verbose guidance injects more noise into the agent's context window at every PRM interval. Closer to 0% is better. | Closer to 0% |
| **BLEU-1** | Unigram overlap between the model's guidance text and Claude's. A proxy for whether the model is giving similar advice to Claude, not just similar length. | Higher |

---

## Rankings

| Rank | Model | Format Compliance | Severity Alignment | Over-Intervention | Under-Intervention | On-Track False Positive | Macro F1 | Guidance Length vs GT | BLEU-1 |
|------|-------|:-----------------:|:------------------:|:-----------------:|:-----------------:|:-----------------------:|:--------:|:---------------------:|:------:|
| 6 | Qwen3-8B (Base) | 68.5% | 35.1% | 13.4% | 22.7% | 17.2% | 0.321 | -50% | 0.207 |
| 5 | Qwen3-8B (Base+Think) | 67.2% | 35.2% | 21.0% | **13.7%** | 28.2% | 0.370 | -48% | 0.237 |
| **1** | **Clean-SFT** | 96.7% | 76.0% | **7.0%** | 15.1% | **17.9%** | 0.522 | **+17%** | **0.475** |
| 2 | Clean-SFT-Think | 95.4% | **76.1%** | 9.1% | **14.4%** | 27.2% | **0.527** | +40% | 0.454 |
| 3 | RS-SFT | **98.7%** | 72.5% | 10.2% | 17.2% | 30.3% | 0.498 | +24% | 0.457 |
| 4 | RS-SFT-Think | 96.9% | 70.7% | 10.5% | 17.2% | 33.0% | 0.480 | +32% | 0.431 |

*Bold = best in column. Ranking by composite sum of per-metric ranks across all 8 dimensions.*

---

## One-line takeaways per model

- **Qwen3-8B (Base)** — 30% unparseable, 35% alignment. Unreliable format, low alignment. Not production-viable.
- **Qwen3-8B (Base+Think)** — Same format issues. Thinking adds +0.049 F1 and reduces under-intervention (22.7% → 13.7%) but doubles over-intervention. Still not production-viable.
- **Clean-SFT** — Best overall. Lowest over-intervention, lowest on-track false positives among SFT models, guidance closest to Claude's style. Use this.
- **Clean-SFT-Think** — Marginally better detection F1 (+0.005) but 27% on-track false positive rate and 40% longer guidance. Not worth it for most use cases.
- **RS-SFT** — Best format compliance (99%) but 30% false intervention on correctly-working agents. Rejection-sampling data introduced noisier training signal.
- **RS-SFT-Think** — Worst on-track FPI (33%) and weakest BLEU. No clear advantage over Clean-SFT.
