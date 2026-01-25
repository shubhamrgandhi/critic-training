python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/prompt_efficient_v2_dev_Qwen3-Coder-30B-A3B-Instruct"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/prompt_efficient_v2_dev_Qwen3-Coder-30B-A3B-Instruct/policy_v2"

python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/prompt_efficient_v3_dev_Qwen3-Coder-30B-A3B-Instruct"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/prompt_efficient_v3_dev_Qwen3-Coder-30B-A3B-Instruct/policy_v2"

# python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/prompt_efficient_dev_cwm"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/prompt_efficient_dev_cwm/policy_v2"

# python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/prompt_efficient_dev_cwm-sft"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/prompt_efficient_dev_cwm-sft/policy_v2"

# python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/prompt_efficient_dev_Devstral-Small-2507"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/prompt_efficient_dev_Devstral-Small-2507/policy_v2"

# python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/prompt_efficient_dev_SWE-agent-LM-32B"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/prompt_efficient_dev_SWE-agent-LM-32B/policy_v2"

# python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/prompt_efficient_dev_Qwen25-Coder-32B-Instruct"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/prompt_efficient_dev_Qwen25-Coder-32B-Instruct/policy_v2"


python analyze_wastage_both.py --setting "prompt_efficient_v2"


python analyze_wastage_both.py --setting "prompt_efficient_v3"

# python analyze_wastage.py --inputs "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/edit_obs_dev_Devstral-Small-2507/policy_v2"

# python analyze_wastage.py --inputs "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/edit_obs_dev_cwm/policy_v2"

# python analyze_wastage.py --inputs "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/edit_obs_dev_SWE-agent-LM-32b/policy_v2"

# python analyze_wastage.py --inputs "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/edit_obs_dev_Qwen3-Coder-30b/policy_v2"

# python analyze_wastage.py --inputs "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/edit_obs_dev_Qwen25-Coder-32B-Instruct/policy_v2"

# python iaa.py

# python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/base_dev_cwm-sft"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/base_dev_cwm-sft/policy_v2"

# python analyze_wastage.py --inputs "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/base_dev_cwm-sft/policy_v2"

# python run-judge-policy-majority-vote.py --results-dir "/usr0/home/srgandhi/tool-overuse/results/base_dev_cwm-pretrain"  --output-dir "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/base_dev_cwm-pretrain/policy_v2"

# python analyze_wastage.py --inputs "/usr0/home/srgandhi/tool-overuse/judge_analysis_majority_vote_k_5/base_dev_cwm-pretrain/policy_v2"
