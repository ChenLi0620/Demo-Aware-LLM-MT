OUTPUT_DIR=${1:-"/UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA/new-r-random-5shot-c0-lora-300/subtitles/"}
TEST_PAIRS=${2:-"de-en"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
export CUDA_VISIBLE_DEVICES=1
python \
    run_llmmt_select.py \
    --model_name_or_path /UNICOMFS/hitsz_mzhang_1/lc/alma-checkpoints/checkpoints/ALMA-7B-R \
    --do_predict \
    --low_cpu_mem_usage True \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path /UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA/human_written_data_domain_subtitles \
    --few_shot_eval_path /UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA/human_written_data_domain_subtitles/new_shots \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --use_peft \
    --peft_model_id  /UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA_lc/alma-7b-parallel-ft_lc_dp_context_r_checkpoint/checkpoint-300/adapter_model \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 1600 \
    --fp16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model 