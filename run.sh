source activate /opt/conda/envs/handbook

accelerate launch --config_file ./configs/zero3.yaml \
    --num_processes=8 \
    grpo_demo.py \
    --wandb_entity="mhong-university-of-minnesota" \
    --model_name="meta-llama/Llama-3.2-1B-Instruct"

# --model_name="Qwen/Qwen2.5-1.5B-Instruct"