# Suppress Singularity/Apptainer mount warnings from system config
export APPTAINER_BIND=""
export SINGULARITY_BIND=""
export APPTAINER_NO_MOUNT="hostfs"
export SINGULARITY_NO_MOUNT="hostfs"

mini-extra swebench-single \
    --config "../mini-swe-agent/configs/swebench_singularity_edit_obs_final_only_0_Devstral-Small-2507.yaml" \
    --subset lite \
    --split dev \
    --output "../results_single/prompt_base_single_dev_Devstral-Small-2507_pvlib__pvlib-python-1072" \
    -i pvlib__pvlib-python-1072
