#!/bin/bash

# CKPT directory
method="sage"
model_name="Qwen2.5-7B-Instruct"
experiment_name="${method}_${model_name}"
ckpts_dir="./outputs/${experiment_name}"
data_path="baohao"

# Configuration
K=16  # Number of samples to generate per prompt, used for averaging for a stable metric
GPUS=(0 1 2 3 4 5 6 7)

# Model and dataset arrays
models=()
for step in $(seq 50 50 500); do
    models+=("${ckpts_dir}/global_step_${step}")
done
datasets=("aime24" "aime25" "amc23" "math500" "minerva_math" "olympiadbench" "gpqa" "mmlu_pro")

# Loop through models and datasets
for model_name in "${models[@]}"; do
    # Convert model to huggingface format
    echo "Converting model: $model_name to huggingface format..."
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${model_name}/actor \
        --target_dir ${model_name}/merged

    echo "Testing model: $model_name"
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"

        # Create model/dataset specific output directory
        output_dir=${model_name}/${dataset}
        mkdir -p ${output_dir}
        echo "Output directory: $output_dir"

        # Generate data in parallel
        echo "Starting parallel data generation..."
        for ((i=0; i<${#GPUS[@]}; i++)); do
            CUDA_VISIBLE_DEVICES=$i python3 eval/gen_data.py \
                --local_index ${i} \
                --my_world_size ${#GPUS[@]} \
                --model_name_or_path ${model_name}/merged \
                --max_input_length 16384 \
                --max_new_tokens 8192 \
                --temperature 0.6 \
                --top_p 0.95 \
                --output_dir ${output_dir} \
                --K $K \
                --dataset_name_or_path ${data_path}/${dataset} &
        done

        # Wait for all parallel processes to complete
        wait
        echo "Data generation completed."

        # Merge the generated data
        echo "Merging data..."
        python3 eval/merge_data.py \
            --base_path ${output_dir} \
            --output_dir ${output_dir}/merged_data.jsonl \
            --num_datasets ${#GPUS[@]}

        if [ $? -ne 0 ]; then
            echo "Error: Failed to merge data for $model_name on $dataset"
            continue
        fi

        # Compute scores
        echo "Computing scores..."
        python3 eval/compute_score.py \
            --dataset_path ${output_dir}/merged_data.jsonl \
            --use_oat_grader \
            --record_path ${output_dir}/record_oat.txt

        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute scores for ${model_name} on ${dataset}"
            continue
        fi

        echo "Completed evaluation for ${model_name} on ${dataset}"
        echo "Results saved to: ${output_dir}/record_oat.txt"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"    