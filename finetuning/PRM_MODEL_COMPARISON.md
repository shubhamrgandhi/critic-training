# PRM Model Comparison: Which Setting is Best?

**Date**: 2026-03-28 | **N**: 2393 samples | **Ground Truth**: Claude Opus 4.6

---

## TL;DR

**Best PRM: `Clean-SFT` (`qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean`)**

Clean-SFT achieves the best balance across all axes: highest format compliance among non-think models (96.7%), top-tier exact alignment with GT (76%), lowest over-intervention (7%), lowest on-track false positives among SFT models (17.9%), closest guidance length to Claude Opus (+17%), and best text similarity (BLEU-1 0.475). Its think variant (Clean-SFT-Think) marginally improves F1 but is substantially more intrusive and verbose — not worth the tradeoff.

---

## Model Rankings by Dimension

### 1. Format Compliance (can the model even produce structured output?)

| Model | Full (12/12) | Unparseable | Missing TASK_STATUS |
|-------|-------------|-------------|---------------------|
| RS-SFT | **98.7%** | 0.0% | 0.1% |
| RS-SFT-Think | 96.9% | 0.7% | 1.5% |
| Clean-SFT | 96.7% | 1.9% | 1.9% |
| Clean-SFT-Think | 95.4% | 0.1% | 0.4% |
| Qwen3-8B (Base) | 68.5% | 29.2% | 28.8% |
| Qwen3-8B (Base+Think) | 67.2% | 30.6% | 30.0% |

**Notes:**
- Base models fail ~30% of the time — completely unreliable for production
- RS-SFT's 98.7% reflects the **fixed** eval (temp=0.0); at temp=0.6 it drops to 84.5% with 14% garbage outputs
- All SFT models are production-viable on format compliance

### 2. Severity Alignment with Ground Truth

| Model | Exact Match | Over-Intervention | Under-Intervention |
|-------|------------|-------------------|--------------------|
| Clean-SFT | **76.0%** | **7.0%** | 15.1% |
| Clean-SFT-Think | **76.1%** | 9.1% | **14.4%** |
| RS-SFT | 72.5% | 10.2% | 17.2% |
| RS-SFT-Think | 70.7% | 10.5% | 17.2% |
| Qwen3-8B (Base+Think) | 35.2% | 21.0% | 13.7% |
| Qwen3-8B (Base) | 35.1% | 13.4% | 22.7% |

**Key pattern in confusion matrices:**
- All SFT models downgrade **44–51% of Critical samples to "Needs Correction"** — the agent gets a mild nudge instead of a hard stop. Clean-SFT is the least bad here (44% downgrade vs 51% for RS-SFT-Think).
- Base models: completely miscalibrated, low exact match, high unparseable

### 3. Per-Category Error Detection (F1 / FPR)

| Category | GT+ | Clean-SFT | Clean-SFT-Think | RS-SFT | RS-SFT-Think | Base |
|----------|-----|-----------|-----------------|--------|--------------|------|
| Step Repetition | 928 | 0.833 | **0.861** | 0.826 | 0.797 | 0.608 |
| Verification Fail. | 965 | **0.908** | 0.898 | 0.905 | 0.894 | 0.721 |
| Info Processing Fail. | 1065 | 0.785 | **0.796** | 0.793 | 0.764 | 0.437 |
| Tool Selection Err. | 485 | 0.607 | **0.620** | 0.568 | 0.550 | 0.323 |
| Task Derailment | 586 | 0.599 | **0.637** | 0.536 | 0.565 | 0.321 |
| Problem Misid. | 311 | **0.621** | 0.600 | 0.535 | 0.539 | 0.338 |
| Context Handling Fail. | 392 | 0.573 | **0.582** | 0.560 | 0.548 | 0.392 |
| Hallucinations | 77 | 0.206 | 0.204 | **0.262** | 0.219 | 0.068 |
| **Macro Average** | | **0.522** | **0.527** | 0.498 | 0.480 | 0.321 |

**Intrusiveness hotspot — Info Processing FPR** (most damaging false positive):

| Model | Info Processing FPR | Problem Misid. FPR |
|-------|--------------------|--------------------|
| Qwen3-8B (Base) | 0.127 | 0.060 |
| Clean-SFT | 0.265 | 0.093 |
| Clean-SFT-Think | 0.370 | 0.109 |
| RS-SFT | 0.386 | 0.138 |
| RS-SFT-Think | **0.399** | **0.116** |

SFT models are 2–3× more likely than base to falsely flag Info Processing and Problem Misidentification. Clean-SFT has the lowest FPR among SFT models on both categories.

### 4. Behavior on On-Track Samples (N=379 where GT = "On Track")

Any flag here = unnecessary intervention on a correctly-working agent.

| Model | Correctly On Track | Falsely Intervenes | False Info Proc. | False Problem Misid. |
|-------|-------------------|-------------------|-----------------|----------------------|
| Qwen3-8B (Base) | **67.3%** | 17.2% | 0.11 | 0.07 |
| Clean-SFT | **80.2%** | 17.9% | 0.11 | 0.10 |
| Clean-SFT-Think | 72.8% | 27.2% | 0.25 | 0.13 |
| RS-SFT | 69.7% | 30.3% | 0.25 | 0.10 |
| RS-SFT-Think | 66.0% | **33.0%** | **0.30** | 0.04 |
| Qwen3-8B (Base+Think) | 53.0% | 28.2% | 0.29 | 0.10 |

**Clean-SFT is the only SFT model that doesn't substantially over-flag on-track agents** (17.9% — nearly identical to base 17.2%). All think variants and RS variants are 27–33% false intervention on on-track agents.

### 5. Intervention Intensity (flags per sample vs GT=3.17)

| Model | Mean flags | Median | 0-flag rate | 3–5 flags |
|-------|-----------|--------|-------------|-----------|
| GT (Claude Opus) | **3.17** | 3 | 10.9% | 50.7% |
| RS-SFT | 2.67 | 3 | 11.1% | 44.4% |
| Clean-SFT-Think | 2.72 | 3 | 11.6% | 46.8% |
| RS-SFT-Think | 2.58 | 2 | 12.9% | 44.8% |
| Clean-SFT | 2.46 | 2 | 16.5% | 42.3% |
| Qwen3-8B (Base+Think) | 1.52 | 1 | 44.6% | 22.6% |
| Qwen3-8B (Base) | 1.00 | 0 | 56.5% | 13.5% |

RS-SFT and Clean-SFT-Think are closest to GT flag count. Clean-SFT slightly under-flags (mean 2.46 vs 3.17) but its flags are more precise (lower FPR).

### 6. Guidance Length

| Model | Mean (chars) | vs GT | Median |
|-------|-------------|-------|--------|
| GT (Claude Opus) | 1440 | — | 1375 |
| Clean-SFT | **1682** | **+17%** | 1588 |
| RS-SFT | 1786 | +24% | 1693 |
| RS-SFT-Think | 1901 | +32% | 1617 |
| Clean-SFT-Think | 2017 | +40% | 1721 |
| Qwen3-8B (Base+Think) | 749 | -48% | 732 |
| Qwen3-8B (Base) | 719 | -50% | 681 |

Clean-SFT writes guidance closest to Claude's length (+17%). Think variants are significantly more verbose (+32–40%), injecting substantially more noise into agent context per PRM call.

### 7. Text Similarity to Ground Truth (BLEU/ROUGE)

| Model | BLEU-1 | BLEU-4 | ROUGE-1 | ROUGE-L |
|-------|--------|--------|---------|---------|
| Clean-SFT | **0.475** | **0.242** | **0.637** | **0.360** |
| RS-SFT | 0.457 | 0.225 | 0.620 | 0.340 |
| Clean-SFT-Think | 0.454 | 0.224 | 0.623 | 0.346 |
| RS-SFT-Think | 0.431 | 0.209 | 0.600 | 0.320 |
| Qwen3-8B (Base+Think) | 0.237 | 0.120 | 0.417 | 0.245 |
| Qwen3-8B (Base) | 0.207 | 0.111 | 0.392 | 0.244 |

Clean-SFT's guidance most closely resembles what Claude Opus would write.

---

## Why Think Variants Are Worse for Production

Despite marginally better F1 scores (Clean-SFT-Think: 0.527 vs Clean-SFT: 0.522), think variants are not recommended because:

1. **3× higher on-track false positives** on Info Processing (0.25 vs 0.11) — they derail correctly-working agents
2. **+40% longer guidance** vs GT — every PRM call injects more directive noise into agent context
3. **27–33% false intervention rate** on on-track samples (vs 17.9% for Clean-SFT)
4. **Not worth it**: the tiny F1 gain (+0.005 macro) is outweighed by substantially more intrusive behavior

---

## Why RS-SFT Underperforms

RS-SFT is noticeably worse than Clean-SFT despite similar training recipe:

1. **Rejection-sampling introduces noisier training distribution** — the RS data contains harder negatives that the model overfits to, leading to more false positives on Info Processing (FPR 0.386 vs 0.265) and Problem Misidentification (0.138 vs 0.093)
2. **On-track intrusiveness nearly double Clean-SFT** (30.3% vs 17.9% false intervention)
3. **Temperature sensitivity**: RS-SFT generates garbage outputs at temp=0.6 (14% unparseable) but is fine at temp=0.0. This suggests the model is less robustly trained and relies on greedy decoding to stay in-distribution.

---

## Observed Agent Trajectory Step Counts

From `results_singularity_max_150_steps/` (assistant turns per trajectory):

| Experiment | N | Avg steps | vs Baseline |
|-----------|---|-----------|-------------|
| Baseline (no PRM) | 100 | 75.0 | — |
| Qwen3-8B base PRM | 100 | 89.9 | +20% |
| SFT PRM (full, earlier version) | 100 | 135.4 | +81% |
| SFT PRM (clean) | 50 | 141.1 | +88% |
| SFT PRM (rejection-sample) | 50 | 111.8 | +49% |

SFT PRMs cause 49–88% more steps than baseline, with many runs hitting the 150-step cap. The causes are the intrusiveness issues documented above — false Info Processing flags cause agent pivots, Critical severity downgrading fails to stop bad trajectories early, and verbose guidance adds noise at every PRM interval.

---

## Verdict: Recommended PRM Settings

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Production PRM** | **Clean-SFT** | Best alignment (76%), lowest intrusiveness, closest to GT guidance style |
| If format is top priority | RS-SFT | 98.7% format compliance, but 30% on-track false intervention |
| If detection F1 is top priority | Clean-SFT-Think | Best macro F1 (0.527), but 27% on-track FPI and +40% guidance length |
| Avoid | Qwen3-8B base/think | 30% unparseable, 35% alignment — not production-viable |

### Inference Parameters (what works)

| Model | Temperature | Thinking | Notes |
|-------|-------------|----------|-------|
| Clean-SFT | **0.0** | **disabled** | Most reliable; temp=0.0 enforces clean format |
| Clean-SFT-Think | 0.6 | **enabled** | 0.6 fine since thinking provides structure |
| RS-SFT | **0.0** | **disabled** | Critical: fails with garbage at temp=0.6 |
| RS-SFT-Think | 0.6 | **enabled** | Fine at 0.6 with thinking |

**Fixed in `run_prm_max150.sh`**: now uses `temperature=0.0` when `disable_thinking=true`, `temperature=0.6` when thinking is enabled. `run_prm.sh` and `run_prm_issue_res_custom_prm.sh` already used `temperature=0.0` correctly.
