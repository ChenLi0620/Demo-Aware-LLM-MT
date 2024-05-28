OUTPUT_DIR=${1:-"./result"}
pairs=${2:-"de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/test_cache_data_r/"
export TRANSFORMERS_CACHE=".cache/models/"
export MASTER_PORT=9902
LORA_RANK=${3:-"16"}
train_path=run_llmmt.py
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $MASTER_PORT \
    ${train_path} \
    --deepspeed /UNICOMFS/hitsz_mzhang_1/lc/parrot/train/deepspeed_config_zero2.json \
    --model_name_or_path /UNICOMFS/hitsz_mzhang_1/lc/alma-checkpoints/checkpoints/ALMA-7B-R \
    --mmt_data_path ./human_written_data/ \
    --use_peft \
    --lora_rank ${LORA_RANK} \
    --do_train \
    --language_pairs ${pairs} \
    --low_cpu_mem_usage \
    --bf16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 4 \
    --logging_strategy steps \
    --logging_steps 25 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 200 \
    --max_source_length 1100 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to none \
