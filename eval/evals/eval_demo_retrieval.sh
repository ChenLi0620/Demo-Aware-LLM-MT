OUTPUT_DIR=${1:-""}
TEST_PAIRS=${2:-"de-en"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
export CUDA_VISIBLE_DEVICES=1
python \
    run_llmmt_select.py \
    --model_name_or_path ./checkpoints/ALMA-7B \
    --do_predict \
    --low_cpu_mem_usage True \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path ./data/data/eval/it \
    --few_shot_eval_path ./data/data/eval/it/new_shots \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --use_peft \
    --peft_model_id  ./checkpoints/Demo_LoRA \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 1024 \
    --fp16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model 
