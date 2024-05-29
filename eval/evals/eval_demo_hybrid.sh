OUTPUT_DIR=${1:-""}
TEST_PAIRS=${2:-"zh-en"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
#export CUDA_VISIBLE_DEVICES=0
python \
    run_llmmt_ol_gf.py \
    --model_name_or_path ./checkpoints/ALMA-7B \
    --do_predict \
    --low_cpu_mem_usage True \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path ./data/data/eval/gf \
    --few_shot_eval_path ./data/data/eval/gf/shots \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 1024 \
    --fp16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model
