OUTPUT_DIR=${1:-"//UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA/result/gf/"}
TEST_PAIRS=${2:-"zh-en"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
#export CUDA_VISIBLE_DEVICES=6
python \
    run_llmmt_ol_gf.py \
    --model_name_or_path /UNICOMFS/hitsz_mzhang_1/lc/alma-checkpoints/checkpoints/ALMA-7B-R \
    --do_predict \
    --low_cpu_mem_usage True \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path /UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA/human_written_data_lc/gf/test/ \
    --few_shot_eval_path /UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA/doc/gf/TEST_1/select_shots \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 1250 \
    --fp16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model