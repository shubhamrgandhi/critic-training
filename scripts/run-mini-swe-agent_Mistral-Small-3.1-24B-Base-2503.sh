mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_Mistral-Small-3.1-24B-Base-2503.yaml" \
    --subset lite \
    --split dev \
    --workers 8 \
    --shuffle \
    --output "/usr0/home/srgandhi/tool-overuse/results/base_dev_Mistral-Small-3.1-24B-Base-2503"

mini-extra swebench \
    --config "/usr0/home/srgandhi/tool-overuse/mini-swe-agent/configs/swebench_Mistral-Small-3.1-24B-Base-2503.yaml" \
    --subset verified \
    --split test \
    --workers 8 \
    --shuffle \
    --output "/usr0/home/srgandhi/tool-overuse/results/base_Mistral-Small-3.1-24B-Base-2503"
