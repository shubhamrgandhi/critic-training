
# mini-extra swebench \
#     --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml" \
#     --subset verified \
#     --split test \
#     --workers 16 \
#     --shuffle \
#     --output "/usr0/home/srgandhi/tool-overuse/results/base_3_Qwen3-Coder-30B-A3B-Instruct"

mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml" \
    --subset verified \
    --split test \
    --workers 16 \
    --shuffle \
    --output "/usr0/home/srgandhi/tool-overuse/results/base_0_Qwen3-Coder-30B-A3B-Instruct"

mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml" \
    --subset verified \
    --split test \
    --workers 16 \
    --shuffle \
    --slice :100 \
    --output "/usr0/home/srgandhi/tool-overuse/results/base_1_Qwen3-Coder-30B-A3B-Instruct"

mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml" \
    --subset verified \
    --split test \
    --workers 16 \
    --shuffle \
    --slice :100 \
    --output "/usr0/home/srgandhi/tool-overuse/results/base_2_Qwen3-Coder-30B-A3B-Instruct"
    
mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_edit_obs_Qwen3-Coder-30B-A3B-Instruct.yaml" \
    --subset verified \
    --split test \
    --workers 16 \
    --shuffle \
    --slice :100 \
    --output "/usr0/home/srgandhi/tool-overuse/results/edit_obs_0_Qwen3-Coder-30B-A3B-Instruct"
    
mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_edit_obs_Qwen3-Coder-30B-A3B-Instruct.yaml" \
    --subset verified \
    --split test \
    --workers 16 \
    --shuffle \
    --slice :100 \
    --output "/usr0/home/srgandhi/tool-overuse/results/edit_obs_1_Qwen3-Coder-30B-A3B-Instruct"
    
mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_edit_obs_Qwen3-Coder-30B-A3B-Instruct.yaml" \
    --subset verified \
    --split test \
    --workers 16 \
    --shuffle \
    --slice :100 \
    --output "/usr0/home/srgandhi/tool-overuse/results/edit_obs_2_Qwen3-Coder-30B-A3B-Instruct"    

# =================================================================

# mini-extra swebench \
#     --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml" \
#     --subset lite \
#     --split dev \
#     --workers 16 \
#     --shuffle \
#     --output "/usr0/home/srgandhi/tool-overuse/results/base_0_dev_Qwen3-Coder-30B-A3B-Instruct"
    
# mini-extra swebench \
#     --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml" \
#     --subset lite \
#     --split dev \
#     --workers 16 \
#     --shuffle \
#     --output "/usr0/home/srgandhi/tool-overuse/results/base_1_dev_Qwen3-Coder-30B-A3B-Instruct"
    
# mini-extra swebench \
#     --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_base_Qwen3-Coder-30B-A3B-Instruct.yaml" \
#     --subset lite \
#     --split dev \
#     --workers 16 \
#     --shuffle \
#     --output "/usr0/home/srgandhi/tool-overuse/results/base_2_dev_Qwen3-Coder-30B-A3B-Instruct"
    
# mini-extra swebench \
#     --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_edit_obs_Qwen3-Coder-30B-A3B-Instruct.yaml" \
#     --subset lite \
#     --split dev \
#     --workers 16 \
#     --shuffle \
#     --output "/usr0/home/srgandhi/tool-overuse/results/edit_obs_0_dev_Qwen3-Coder-30B-A3B-Instruct"
    
# mini-extra swebench \
#     --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_edit_obs_Qwen3-Coder-30B-A3B-Instruct.yaml" \
#     --subset lite \
#     --split dev \
#     --workers 16 \
#     --shuffle \
#     --output "/usr0/home/srgandhi/tool-overuse/results/edit_obs_1_dev_Qwen3-Coder-30B-A3B-Instruct"
    
# mini-extra swebench \
#     --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_edit_obs_Qwen3-Coder-30B-A3B-Instruct.yaml" \
#     --subset lite \
#     --split dev \
#     --workers 16 \
#     --shuffle \
#     --output "/usr0/home/srgandhi/tool-overuse/results/edit_obs_2_dev_Qwen3-Coder-30B-A3B-Instruct"