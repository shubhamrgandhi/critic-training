# critic-training

Code for training and evaluating a step-level critic (PRM) for software-engineering agents on SWE-bench Verified and R2E-Gym, using mini-swe-agent.

- **Just want to run something fast?** → [QUICKSTART.md](QUICKSTART.md) — clone → install → serve → run, in ~5 commands.
- **Need the full picture?** → [HANDOVER.md](HANDOVER.md) — directory map, where data lives on Babel, the LiteLLM-gateway path for reproducing without AWS, naming conventions, and end-to-end commands for both training and inference.

The trained critic is on Hugging Face: [`shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn`](https://huggingface.co/shubhamrgandhi/qwen3-8b-full-sft-prm-r2egym-swebench-k5-cwm-plus-qwen-opus-distill-32k-multiturn).
