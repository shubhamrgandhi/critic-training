# Issue: SFT on Qwen3-8B Destroys Native Thinking/Reasoning Ability

## Problem

After full SFT of Qwen3-8B on PRM feedback data, the model loses its native `<think>...</think>` reasoning ability. All finetuned checkpoints exhibit this — the model outputs reasoning-like text inline (without `<think>` opening tag, with a stray `</think>` closing tag) or no reasoning at all, depending on the template used.

## Verified Behavior

Tested on babel-x5-28 with vLLM 0.15.1, prompt: "What is 2+2? Think step by step."

| Model | `<think>` present | `</think>` present | Notes |
|---|---|---|---|
| **Qwen/Qwen3-8B** (base) | Yes | Yes | Proper `<think>reasoning</think>response` format |
| **sft-rejection-sample** | No | Yes | Reasoning dumped into content without opening tag |
| **sft-clean** | No | Yes | Same broken behavior |
| **sft-noisy** | No | Yes | Same broken behavior |

## Root Cause

The training data (PRM feedback from Claude Opus) contains only the final PRM response — no chain-of-thought reasoning inside `<think>` tags. Combined with the LlamaFactory template, this teaches the model to skip thinking.

## What We Tried / Investigated

### LlamaFactory Template Options

1. **`template: qwen3_nothink`** (what was used for all existing checkpoints)
   - Removes `<think>` tokens entirely from the chat template
   - Model never sees thinking delimiters during training
   - Result: Model completely unlearns thinking format

2. **`template: qwen3` + `enable_thinking: false`** (fast-thinking mode, PR #7923)
   - LlamaFactory adds empty `<think>\n</think>` to the template
   - Loss is masked on `<think></think>` tokens (no gradient from them)
   - BUT: the model still sees empty `<think></think>` in every training example, so attention context learns to expect empty reasoning
   - Result: Likely teaches model to produce empty thinking (not tested yet)

3. **`template: qwen3` + `enable_thinking: true`** (slow-thinking mode)
   - Loss IS computed on `<think>` content
   - Requires actual CoT reasoning in training data
   - Not viable without synthetic reasoning generation

### Serving Configuration

- vLLM should be served with `--enable-reasoning --reasoning-parser deepseek_r1` for Qwen3
- This makes vLLM parse `<think>...</think>` into `message.reasoning_content` (separate field)
- Without these flags, everything goes into `message.content` and `reasoning_content` is null
- Moot point if the model doesn't produce `<think>` tags after SFT

### Other Findings

- Qwen3 requires `temperature >= 0.6` for thinking mode; `temperature=0.0` causes repetition loops (e.g., `"SPECIES: HUMAN (Bipedal, Bipedal, Bipedal..."`)
- `extra_body: {"enable_thinking": False}` is ignored by vLLM 0.15.x; use `chat_template_kwargs` instead
- LlamaFactory README: "use the SAME template in training and inference"

## The Fundamental Problem

There is no way to preserve Qwen3's native reasoning through SFT without reasoning content in the training data. All three template options either destroy thinking, teach empty thinking, or require CoT data we don't have.

## Possible Solutions (Not Yet Implemented)

1. **Generate synthetic CoT**: Run each training prompt through base Qwen3-8B or another model to produce reasoning before the PRM response. Use as training targets with `template: qwen3` + `enable_thinking: true`. Expensive but principled.

2. **LoRA instead of full SFT**: Fine-tune with LoRA on a small subset of weights. May preserve more of the base model's behavior including thinking. Not tested.

3. **Test fast-thinking mode empirically**: Option 2 above (`qwen3` + `enable_thinking: false`) might still produce *some* reasoning in practice even if degraded. Files are ready to run:
   - Config: `qwen3_8b_prm_full_sft_l40s_train_think.yaml`
   - Script: `run_qwen3_8b_prm_full_sft_l40s_think.sh`
   - Data: `prm_sft_data_opus_distill_full_feedback_history_32k_rejection-sample` (1303 samples)
   - Epochs: 1
   - Output: `qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample_think`

4. **Accept no thinking**: Use `qwen3_nothink`, serve without `--enable-reasoning`, and rely on the PRM response quality alone. This is what current checkpoints do.

## Files Changed During Investigation

- `eval_generate_responses.py` — updated to extract reasoning from responses (handles missing `<think>` tag), uses `chat_template_kwargs` for vLLM, captures reasoning in `"reasoning"` field of responses.jsonl
- `qwen3_8b_prm_full_sft_l40s_train_think.yaml` — NEW: training config with `template: qwen3` + `enable_thinking: false`
- `run_qwen3_8b_prm_full_sft_l40s_think.sh` — NEW: run script for think-mode training
- `checkpoints.json` — NEW: checkpoint registry for eval generation
- `test_thinking.py` — NEW: quick test script to check if a served model produces thinking tags

## Existing Checkpoints (all use qwen3_nothink, no thinking)

- `qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_rejection-sample` — rejection-sampled data, 3 epochs
- `qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_clean` — clean data
- `qwen3-8b-full-sft-prm-opus-distill-32k-lr5e6_noisy` — noisy data

## References

- LlamaFactory PR #7923: [optimize qwen3 loss computation](https://github.com/hiyouga/LLaMA-Factory/pull/7923)
- LlamaFactory Issue #7905: [Qwen3 thinking dataset handling](https://github.com/hiyouga/LLaMA-Factory/issues/7905)
- LlamaFactory Issue #9090: [qwen3_nothink SFT output anomaly](https://github.com/hiyouga/LLaMA-Factory/issues/9090)
- Qwen3 HuggingFace: recommends `vllm serve --enable-reasoning --reasoning-parser deepseek_r1`
