OUTPUT_DIR=${1:-"/data/lc/guangzhou/ALMA/ALMA/ALMA/llama2_base/it/"}
TEST_PAIRS=${2:-"de-en"}
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"
#export CUDA_VISIBLE_DEVICES=1
python \
    run_llmmt.py \
    --model_name_or_path /data/llx/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9 \
    --do_predict \
    --low_cpu_mem_usage True \
    --language_pairs ${TEST_PAIRS} \
    --mmt_data_path /data/lc/guangzhou/ALMA/ALMA/ALMA/human_written_data_domain_it \
    --per_device_eval_batch_size 1 \
    --output_dir ${OUTPUT_DIR} \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 400 \
    --fp16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --multi_gpu_one_model 