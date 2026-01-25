mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_cwm-pretrain.yaml" \
    --subset lite \
    --split dev \
    --workers 8 \
    --shuffle \
    --output "/usr0/home/srgandhi/tool-overuse/results/base_dev_cwm-pretrain"

mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_cwm-pretrain.yaml" \
    --subset verified \
    --split test \
    --workers 8 \
    --shuffle \
    --output "/usr0/home/srgandhi/tool-overuse/results/base_cwm-pretrain"