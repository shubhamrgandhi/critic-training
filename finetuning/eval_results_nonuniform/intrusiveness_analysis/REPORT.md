# PRM Intrusiveness Analysis Report

**Date**: 2026-03-27  |  **N**: 2393 samples  |  **Ground Truth**: Claude Opus PRM responses

**Models compared**:
- **Qwen3-8B (Base)**: Untrained Qwen3-8B used as PRM
- **RS-SFT**: Qwen3-8B finetuned via SFT on rejection-sampled Claude Opus PRM distillation data
- **RS-SFT-Think**: Same as RS-SFT but with thinking/reasoning enabled during generation

**Core question**: Why do RS-SFT and RS-SFT-Think lead to substantially longer agent trajectories (more steps) compared to both the untrained Base PRM and Claude Opus PRM?

---

## Quick Guide: What to Focus On

| Priority | Plot | What it shows |
|----------|------|---------------|
| 1 | `03_task_status_confusion.png` | RS-SFT downgrades 49% of Critical to Needs Correction — delays hard stops |
| 2 | `05a_fpr_by_category.png` | RS-SFT has *higher* FPR than Base on Info Processing (0.39 vs 0.28) and Problem Misid (0.14 vs 0.07) |
| 3 | `07c_on_track_fpr.png` | When agent is doing fine, RS-SFT falsely flags Info Processing 22% and Problem Misid 13% of the time |
| 4 | `04_alignment.png` | RS-SFT: 73% severity match, but 17% under-intervention (almost all on Critical samples) |
| 5 | `08_guidance_length.png` | RS-SFT writes 24% longer guidance than Claude; RS-SFT-Think is 39% longer |

---

## Executive Summary

RS-SFT improves substantially over Base on format compliance (99% vs 67%), overall severity alignment (73% vs 35%), and macro-F1 (0.50 vs 0.36). However, it introduces a specific failure pattern that likely drives longer trajectories: **it compresses the severity scale toward "Needs Correction"**, simultaneously (a) downgrading Critical samples and (b) over-flagging specific error categories on non-critical samples.

The net effect on the agent:
1. When things are going badly (Critical) → RS-SFT gives a mild nudge instead of a hard stop → agent continues on wrong path
2. When things are going fine (On Track) → RS-SFT falsely flags reasoning errors 30% of the time → agent pivots unnecessarily
3. In both cases, the guidance text is 24-39% longer than Claude's → more noise per PRM injection

Base avoids problem #2 because it's unparseable 31% of the time (effectively abstaining) and under-flags when it does respond. Claude Opus avoids all three by definition (it is the ground truth).

---

## 1. Format Compliance

*Plot: `01_format_compliance.png` | CSV: `01_format_compliance.csv`*

| Model | All 12 categories parsed | Unparseable (0/12) | Missing TASK_STATUS |
|-------|-------------------------|-------------------|-------------------|
| Qwen3-8B (Base) | 66.7% | 31.5% | 31.0% |
| RS-SFT | 98.7% | 0.1% | 0.1% |
| RS-SFT-Think | 96.4% | 1.0% | 1.2% |

The PRM output format requires 12 numbered error categories (each with `DETECTED: Yes/No`), a `TASK_STATUS`, and `OVERALL_GUIDANCE`. RS-SFT learns this format extremely well. Base fails ~31% of the time — these unparseable responses are effectively wasted PRM calls where the agent receives no structured feedback.

**Implication for trajectory length**: Base's 31% unparseable rate actually *helps* it — those are 31% of PRM checkpoints where the agent receives no directive feedback and just continues doing what it's doing. RS-SFT provides structured feedback on 99% of checkpoints, so every PRM call has the potential to redirect the agent.

---

## 2. TASK_STATUS Severity Alignment

*Plots: `02_task_status_dist.png`, `03_task_status_confusion.png` (FOCUS), `04_alignment.png`*

### Overall alignment

| Model | Exact Match | Over-Intervention | Under-Intervention | Unparseable |
|-------|------------|-------------------|-------------------|-------------|
| Qwen3-8B (Base) | 34.6% | 21.1% | 13.3% | 31.0% |
| RS-SFT | 72.5% | 10.2% | 17.2% | 0.1% |
| RS-SFT-Think | 71.0% | 11.2% | 16.6% | 1.2% |

### The "Needs Correction" magnet problem

RS-SFT's status distribution reveals a systematic compression toward "Needs Correction":

| Status | GT (Claude) | Base | RS-SFT | RS-SFT-Think |
|--------|------------|------|--------|-------------|
| On Track | 15.8% | 16.3% | 17.2% | 17.0% |
| **Needs Correction** | **61.6%** | **27.3%** | **65.5%** | **63.4%** |
| Critical | 22.5% | 25.4% | 17.2% | 18.5% |

RS-SFT produces ~65% "Needs Correction" vs GT's 62% — close overall, but this masks a severe problem visible in the confusion matrix:

### Confusion matrix: Critical samples (N=539)

| Model | Correctly says Critical | Downgrades to Needs Correction | Downgrades to On Track |
|-------|------------------------|-------------------------------|----------------------|
| Base | 37.5% | 19.3% | 5.2% |
| **RS-SFT** | **50.6%** | **49.2%** | **0.0%** |
| **RS-SFT-Think** | **54.4%** | **44.2%** | **0.2%** |

**This is the single most important finding.** When the agent is critically off-track, RS-SFT says "Needs Correction" instead of "Critical" in 49% of cases. The agent gets a suggestion to adjust rather than a signal to stop and rethink fundamentally.

On downgraded Critical samples, RS-SFT flags 3.6 categories on average (vs 4.7 on correctly-identified Critical). The most commonly *missed* categories on these downgraded samples are Context Handling (37%), Task Derailment (32%), and Tool Selection (19%) — these are the coordination-level errors that distinguish Critical from Needs Correction.

### Confusion matrix: "Needs Correction" bucket precision

When RS-SFT says "Needs Correction" (N=1568):
- 76.4% are truly Needs Correction ✓
- **16.9% are actually Critical** ← these are the dangerous ones
- 6.7% are actually On Track

So ~1 in 6 of RS-SFT's "Needs Correction" calls is actually a Critical situation where the agent should be stopping.

---

## 3. Per-Category Error Detection

*Plots: `05a_fpr_by_category.png` (FOCUS), `05b_fnr_by_category.png`, `05c_f1_by_category.png`, `05d_fpr_fnr_combined.png`*

*Convention: Positive = DETECTED: Yes (error flagged), Negative = DETECTED: No (no error)*

### F1 scores vs ground truth

| Category | GT+ | Base F1 | RS-SFT F1 | RS-SFT-Think F1 |
|----------|-----|---------|-----------|----------------|
| Task Spec Viol. | 67 | 0.261 | 0.390 | 0.325 |
| Step Repetition | 902 | 0.652 | 0.826 | 0.803 |
| Termination Unaware. | 96 | 0.222 | 0.380 | 0.444 |
| Problem Misid. | 267 | 0.283 | 0.535 | 0.507 |
| Tool Selection Err. | 495 | 0.462 | 0.567 | 0.572 |
| Hallucinations | 74 | 0.180 | 0.252 | 0.163 |
| Info Processing Fail. | 1006 | 0.578 | 0.790 | 0.762 |
| Task Derailment | 579 | 0.371 | 0.536 | 0.570 |
| Goal Deviation | 112 | 0.125 | 0.220 | 0.212 |
| Context Handling Fail. | 371 | 0.401 | 0.566 | 0.584 |
| Verification Fail. | 934 | 0.811 | 0.905 | 0.905 |
| **Macro Average** | | **0.362** | **0.497** | **0.487** |

### Net error direction (FP minus FN): Where does RS-SFT over-flag?

Most categories have net negative (FP < FN), meaning RS-SFT still under-flags. But three categories have **net positive** (RS-SFT flags them MORE than GT does):

| Category | RS-SFT FP | RS-SFT FN | Net | Direction |
|----------|----------|----------|-----|-----------|
| **Problem Misid.** | 265 | 186 | **+79** | OVER-FLAGS |
| **Hallucinations** | 90 | 82 | **+8** | OVER-FLAGS |
| **Verification Fail.** | 180 | 93 | **+87** | OVER-FLAGS |

And one category is nearly balanced but has very high FPR:

| Category | RS-SFT FP | RS-SFT FN | Net | FPR |
|----------|----------|----------|-----|-----|
| **Info Processing Fail.** | 319 | 320 | -1 | **0.388** |

Info Processing is the most consequential: RS-SFT fires 319 false positives while also missing 320 true positives. It's basically **randomly flagging** this category relative to GT — the false positive rate (0.39) is almost as high as the true positive rate (0.79). The signal-to-noise ratio is poor.

### Intrusiveness hotspots — RS-SFT FPR > Base FPR

| Category | Base FPR | RS-SFT FPR | RS-SFT-Think FPR | Impact on agent |
|----------|---------|-----------|-----------------|-----------------|
| **Info Processing Fail.** | 0.276 | **0.388** | **0.420** | Tells agent its reasoning is wrong when it isn't |
| **Problem Misid.** | 0.073 | **0.139** | **0.123** | Agent re-analyzes problem from scratch |
| **Hallucinations** | 0.011 | **0.040** | **0.032** | Agent distrusts its own correct findings |

These are categories where RS-SFT is *more intrusive than not finetuning at all*.

---

## 4. Intervention Intensity

*Plot: `06a_flag_count_dist.png` | CSV: `06_intervention_intensity.csv`*

| Metric | GT (Claude) | Base | RS-SFT | RS-SFT-Think |
|--------|------------|------|--------|-------------|
| Mean flags/sample | 3.12 | 1.51 | 2.67 | 2.60 |
| Median | 3 | 1 | 3 | 2 |
| 0 flags (abstain) | 12.6% | 45.0% | 11.2% | 12.3% |
| 1-2 flags | 25.3% | 28.8% | 38.4% | 38.6% |
| 3-5 flags | 49.6% | 22.0% | 44.4% | 43.8% |
| 6+ flags | 12.5% | 4.2% | 6.0% | 5.3% |

Base abstains 45% of the time (flags zero categories). RS-SFT matches GT's abstention rate (11% vs 13%). However, RS-SFT under-flags in the high-severity bucket: only 6% of RS-SFT responses flag 6+ categories vs 12.5% for GT.

**Under-intervention rate increases with GT severity:**

| GT flags | RS-SFT under-intervention rate |
|----------|-------------------------------|
| 0 flags | 3.7% |
| 1-2 flags | 16.2% |
| 3-5 flags | 17.2% |
| **6+ flags** | **33.1%** |

RS-SFT under-intervenes on a third of the most severe samples. This is the category-level manifestation of the same severity compression seen in TASK_STATUS.

---

## 5. Behavior on "On Track" Samples (N=379)

*Plot: `07c_on_track_fpr.png` (FOCUS) | CSV: `07a_on_track_behavior.csv`*

When GT says the agent is doing fine (On Track):

| Model | Correctly says On Track | Falsely says Needs Correction | Falsely says Critical |
|-------|------------------------|------------------------------|---------------------|
| Base | 46.7% | 26.1% | 6.6% |
| RS-SFT | 69.7% | 27.7% | 2.6% |
| RS-SFT-Think | 64.9% | 31.1% | 2.9% |

RS-SFT is better than Base at recognizing On Track (70% vs 47%), but still falsely intervenes ~30% of the time, and the nature of the false intervention is different:

### False positive breakdown on On Track samples

| Category | Base FP | RS-SFT FP | RS-SFT-Think FP |
|----------|--------|----------|----------------|
| **Info Processing** | 24 (6.3%) | **84 (22.2%)** | **101 (26.6%)** |
| **Problem Misid.** | 11 (2.9%) | **49 (12.9%)** | **41 (10.8%)** |
| Verification | 85 (22.4%) | 34 (9.0%) | 35 (9.2%) |

Base's false positives are dominated by Verification (harmless — the agent just verifies more). RS-SFT's false positives are dominated by Info Processing and Problem Misidentification — these are **reasoning-level accusations** that cause the agent to doubt and redo its own correct work.

When RS-SFT falsely flags Info Processing on On Track samples (N=84), it also co-flags Problem Misidentification 30% of the time. This creates a compound false signal: "you're reasoning wrong AND you misidentified the problem" — which is maximally disruptive to a correctly-working agent.

---

## 6. Guidance Length

*Plot: `08_guidance_length.png` | CSV: `08_guidance_length.csv`*

| Model | Mean Guidance (chars) | vs GT | Breakdown by GT status |
|-------|----------------------|-------|----------------------|
| GT (Claude Opus) | 1440 | -- | OT: 1342, NC: 1344, Crit: 1773 |
| Base | 746 | -48% | OT: 624, NC: 752, Crit: 813 |
| RS-SFT | 1786 | **+24%** | OT: 1598, NC: 1724, Crit: 2089 |
| RS-SFT-Think | 2000 | **+39%** | OT: 1577, NC: 1927, Crit: 2497 |

RS-SFT writes longer guidance than Claude across all severity levels. Notably, on On Track samples where ideally the guidance should be brief ("keep going, you're on track"), RS-SFT writes 1598 chars (19% more than Claude's already-long 1342 chars for On Track).

RS-SFT-Think is even worse: 2497 chars on Critical samples (41% more than Claude). The thinking process seems to generate more verbose output.

---

## 7. Text Similarity (BLEU/ROUGE)

*Plot: `09_bleu_rouge.png` | CSV: `09_bleu_rouge.csv`*

| Metric | Base | RS-SFT | RS-SFT-Think |
|--------|------|--------|-------------|
| BLEU-1 | 0.233 | 0.457 | 0.432 |
| BLEU-4 | 0.118 | 0.225 | 0.210 |
| ROUGE-1 (F) | 0.409 | 0.620 | 0.600 |
| ROUGE-L (F) | 0.241 | 0.340 | 0.320 |

RS-SFT roughly doubles BLEU and increases ROUGE by ~50% vs Base. It learned Claude's *vocabulary and phrasing* well. But text similarity doesn't imply classification correctness — RS-SFT can sound like Claude while making different (wrong) decisions.

---

## 8. Response Length

*Plot: `10_response_length.png` | CSV: `10_response_length.csv`*

| Model | Mean Full Response | Ratio vs GT |
|-------|-------------------|-------------|
| GT (Claude Opus) | 4029 chars | 1.00 |
| Base | 3034 chars | 0.79 |
| RS-SFT | 4644 chars | 1.19 |
| RS-SFT-Think | 5584 chars | 1.41 |

RS-SFT-Think's full PRM responses are 41% longer than Claude's. Since the full PRM output is injected into the agent context every `prm_interval` steps, this means RS-SFT-Think consumes ~5.6KB per feedback injection vs Claude's ~4KB.

---

## 9. Head-to-Head: RS-SFT vs Base

On the 1648 samples where both models produce parseable TASK_STATUS:

| Outcome | Count | % |
|---------|-------|---|
| Both correct | 616 | 37.4% |
| RS-SFT closer to GT | 621 | 37.7% |
| Base closer to GT | 214 | 13.0% |
| Both wrong | 197 | 12.0% |

RS-SFT is better overall (37.7% vs 13.0% wins). But the 214 cases where Base beats RS-SFT are dominated by:
- **GT=Critical, Base=Critical, RS-SFT=Needs Correction** (n=94) — the severity downgrade problem
- **GT=On Track, Base=On Track, RS-SFT=Needs Correction** (n=44) — false intervention on good trajectories

These are exactly the two failure modes that drive longer trajectories.

---

## 10. Effective Intervention Rate

| Model | Intervention rate | Precision of intervention | False interventions |
|-------|------------------|--------------------------|-------------------|
| Base | 76.3% of parseable | 90.2% | 124 |
| RS-SFT | 82.8% | 94.2% | 115 |
| RS-SFT-Think | 82.8% | 93.4% | 129 |

GT intervention rate is 84.2%. RS-SFT's intervention rate (82.8%) is actually close to GT. The problem isn't that RS-SFT intervenes too much — it's that **the quality and severity of interventions is miscalibrated**.

Base intervenes less (76.3%) and with lower precision (90.2%), but its errors are distributed more randomly. RS-SFT's errors are concentrated in specific failure modes (severity downgrading, Info Processing FP) that are particularly harmful to trajectory length.

---

## Root Cause Analysis: Why RS-SFT Leads to Longer Trajectories

### The core problem: Severity compression

RS-SFT learned to generate Claude-like text but failed to learn Claude's **calibration** of severity. It compresses the severity scale toward "Needs Correction":
- 49% of Critical → Needs Correction (should be a hard stop, gets a mild nudge)
- 17% under-intervention overall
- 33% under-intervention on 6+ flag samples

This is likely a training data artifact: "Needs Correction" is 62% of the GT distribution. The model learned that "Needs Correction" is the safe default, and gravitates toward it even when the evidence warrants Critical.

### The secondary problem: Reasoning-category false positives

RS-SFT learned to flag Info Processing (FPR=0.39) and Problem Misidentification (FPR=0.14) at rates higher than Base. These specific categories are maximally disruptive because they attack the agent's *reasoning process* rather than pointing out a specific error. When the PRM says "you're reasoning wrong" or "you misidentified the problem," the agent doesn't just fix a local issue — it re-evaluates its entire approach.

On On Track samples, RS-SFT's false positives are concentrated in Info Processing (22%) and Problem Misid (13%), with 30% co-occurrence. Base's false positives are mostly Verification (22%) — which is much less disruptive ("verify your answer" is not "your reasoning is wrong").

### The tertiary problem: Verbose guidance

RS-SFT writes 24-39% more text per feedback cycle. This is problematic in combination with the above: the agent receives longer, more detailed, but *miscalibrated* guidance. More text means more specific (incorrect) suggestions for the agent to follow.

### How these compound for trajectory length

**Scenario A — Critical situation (22.5% of samples):**
1. Claude says Critical → agent stops and rethinks → short trajectory
2. RS-SFT says Needs Correction (49% of the time) → agent makes a small adjustment → continues on wrong path → needs more steps to either self-correct or get caught at next PRM checkpoint

**Scenario B — On Track situation (15.8% of samples):**
1. Claude says On Track → agent continues → no wasted steps
2. RS-SFT says Needs Correction (28% of the time), often with false Info Processing flag → agent doubts its correct reasoning → pivots or backtracks → wastes steps getting back on track

**Scenario C — Needs Correction situation (61.6% of samples):**
1. Claude says Needs Correction with calibrated guidance → agent adjusts efficiently
2. RS-SFT says Needs Correction (81% correct here) but with 24% longer guidance → agent gets more directives to process, some potentially conflicting due to false category flags

### Why Base doesn't have this problem (despite being worse overall)

Base avoids the trajectory-length problem through a different failure mode:
- 31% of responses are unparseable → effectively abstains → no intervention → agent continues unimpeded
- When it does parse, it under-flags (mean 1.5 flags vs GT 3.1) → weaker intervention signal
- Its false positives are dominated by Verification (harmless) rather than Info Processing (disruptive)

Base is a worse PRM by every metric, but its errors are *benign for trajectory length*. RS-SFT's errors are *actively harmful for trajectory length*.

### Why Claude Opus doesn't have this problem

Claude Opus is the ground truth, so by definition its severity calibration is correct. But beyond that, Claude likely has better calibration because:
- It has stronger reasoning capabilities to distinguish Critical from Needs Correction
- It doesn't have the "Needs Correction" class imbalance bias from SFT training

---

## Potential Mitigations

1. **Upsample Critical training examples**: The model learned "Needs Correction" as a default because it's 62% of the training data. Upsample Critical examples (22.5% → ~33-40%) or use a severity-weighted loss to make Critical/Needs Correction boundary decisions higher-stakes during training.

2. **Category-specific post-processing**: At inference time, suppress Info Processing and Problem Misidentification flags if confidence is below a threshold (e.g., using logprobs). These two categories have the worst FPR and are the most disruptive to trajectory length.

3. **Truncate guidance length**: Cap OVERALL_GUIDANCE at Claude's p75 (~1615 chars) to reduce noise injection.

4. **Two-stage approach**: Use RS-SFT for category detection (where it's good — macro-F1 0.50) but use a separate calibrated classifier or heuristic for TASK_STATUS severity. The severity decision could be derived from the number and type of flagged categories rather than generated directly.

5. **Rejection sample on severity accuracy**: During rejection sampling, prioritize samples where the model correctly distinguishes Critical from Needs Correction, rather than optimizing for overall text similarity.

6. **Differential PRM injection**: Don't inject full PRM feedback on every checkpoint. Only inject when TASK_STATUS is Critical, or when specific high-confidence categories are flagged. This would reduce the noise from false positives on On Track and mild Needs Correction samples.
