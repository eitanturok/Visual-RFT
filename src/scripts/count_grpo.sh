# There's no need for prolonged training. For a dataset with only a few hundred samples, 200 steps should be sufficient.
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_count_countbenchqa_train.txt"

# export DATA_PATH=./share_data/ViRFT_CountBenchQA_train   ### your local dataset downloading from huggingface
export DATA_PATH=eturok/CountBenchQA
export CKPT_PATH=Qwen/Qwen2-VL-2B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_Count   ### save path

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_GRPO_Count \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8
