#!/usr/bin/env python
# coding=utf-8

import logging
import copy
import math
import os
import sys
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    LlamaTokenizer,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from peft import PeftModel, PeftConfig
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback
from datasets import concatenate_datasets, interleave_datasets
from utils_ol.trainer_llmmt import LlmmtTrainer
from utils_ol.utils import LANG_TABLE, load_mmt_dataset, get_preprocessed_data, clean_outputstring, load_tokenizer, load_model, SavePeftModelCallback, get_key_suffix, get_prompt
from utils_ol.arguments import ModelArguments, DataTrainingArguments
from utils_ol.ul2collator import DataCollatorForUL2

logger = logging.getLogger(__name__)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
LANG_TABLE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "is": "Icelandic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ha": "Hausa",
    "ro": "Romanian",
}

## Prefix and suffix for prompt in target language (only from English to target language if the target is non-English)
PREFIX = {
    "de": "Übersetzen Sie dies vom Englischen ins Deutsche:\nEnglisch: ",
    "fr": "Traduisez ceci de l'anglais vers le français :\nAnglais: ",
    "cs": "Přeložte toto z angličtiny do češtiny:\nanglicky: ",
    "is": "Þýddu þetta úr ensku yfir á íslensku:\nEnska: ",
    "zh": "将其从英文翻译成中文：\n英语：",
    "ja": "これを英語から日本語に翻訳してください:\n英語：",
    "ru": "Переведите это с английского на русский:\nАнглийский: ",
    "uk": "Перекладіть це з англійської на українську:\nАнглійська: ",
    "ha": "Fassara wannan daga Turanci zuwa Hausa:\nTuranci: ",
}

SUFFIX = {
    "en": "\nEnglish:",
    "de": "\nDeutsch:",
    "fr": "\nFrançais :",
    "cs": "\nčesky:",
    "is": "\nÍslenska:",
    "zh": "\n中文：",
    "ja": "\n日本語：",
    "ru": "\nРусский:",
    "uk": "\nУкраїнська:",
    "ha": "\nHausa:",
}

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_llmmt", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Get the datasets
    pairs = set(data_args.language_pairs.split(","))
    train_raw_data, valid_raw_data, test_raw_data = None, None, None
    if data_args.mmt_data_path:
        train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(pairs, data_args, model_args, training_args, logger)
    print(test_raw_data)
    if data_args.mono_data_path:
        train_raw_data = load_dataset(
            "json",
            data_files=data_args.mono_data_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
    if data_args.oscar_data_path:
        oscar_langs = data_args.oscar_data_lang.split(",")
        if data_args.interleave_probs:
            interleave_probs = [float(p) for p in data_args.interleave_probs.split(",")]
        else:
            interleave_probs = [1/len(oscar_langs)] * len(oscar_langs)
        oscar_langs = [x for x, _ in sorted(zip(oscar_langs, interleave_probs), key=lambda zippair: zippair[1])]
        interleave_probs = sorted(interleave_probs)
        train_raw_data = []
        for lg in oscar_langs:
            train_raw_data.append(
                load_dataset(
                    data_args.oscar_data_path,
                    lg,
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                )['train']
            )
        train_raw_data = interleave_datasets(train_raw_data, probabilities=interleave_probs, seed=training_args.seed, stopping_strategy="all_exhausted")
    
    # load tokenizer
    set_seed(training_args.seed)
    tokenizer = load_tokenizer(data_args, model_args, training_args, logger)
    if data_args.use_ul2:
        assert data_args.use_prefix_lm, "Must enable use prefix language model"

    shots_eval_dict = {}
    shots_eval_dict_1 = {}
    if data_args.few_shot_eval_path:
        for lg_pair in test_raw_data.keys():
            pair_shot_path = os.path.join(data_args.few_shot_eval_path, f"shots.{lg_pair}.json")
            if not os.path.isfile(pair_shot_path):
                ValueError(f"Make sure the language pair {lg_pair} is in the few shot eval folder!")
            with open(pair_shot_path) as f:
                shots_eval_dict_1[lg_pair] = json.load(f)

    train_datasets, eval_datasets, test_datasets = get_preprocessed_data(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args)
    print(test_datasets)
    metric = evaluate.load("./train/metric/evaluate/metrics/sacrebleu/sacrebleu.py")
    #metric = evaluate.load("sacrebleu")
    # Load model
    model = load_model(data_args, model_args, training_args, tokenizer, logger)
    collate_fn = DataCollatorForUL2(model, tokenizer) if data_args.use_ul2 else default_data_collator
    
    # Initialize our Trainer
    trainer = LlmmtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=[SavePeftModelCallback] if model_args.use_peft else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_state()
        if model_args.use_peft:
            model.save_pretrained(training_args.output_dir) 
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload
    # Prediction
    def get_prompt_few_shot(source_lang, target_lang, ex, idx, source_path, target_path,shots_eval_dict):
        print(shots_eval_dict)
        src_fullname = LANG_TABLE[source_lang]
        tgt_fullname = LANG_TABLE[target_lang]
        prefix = f"Translate this from {src_fullname} to {tgt_fullname}:"
        shot_prompt = ""
        
        lg_pair = f"{source_lang}-{target_lang}"
        lg_shots = shots_eval_dict.get(lg_pair, {})  # 获取特定语言对应的字典
        src_sentence = ex[source_lang]
        shots = lg_shots.get(src_sentence, [])[:2]  # 获取与源句匹配的前2个示例
        shot_prompt0 = ""
        for shot in shots:
            shot_src = shot['source']
            shot_tgt = shot['target']
            shot_prompt0 += f"\n{src_fullname}: " + shot_src + f"\n{tgt_fullname}: " + shot_tgt
        # 根据索引idx读取相应的行
        print(shot_prompt0)
        with open(source_path, 'r', encoding='utf-8') as src_file, open(target_path, 'r', encoding='utf-8') as tgt_file:
            source_lines = src_file.readlines()
            target_lines = tgt_file.readlines()


            # 选择用于 few-shot prompt 的行
            start_line = max(0, idx - 3)
            end_line = max(0, idx)
            for i in range(start_line, end_line):
                shot_src = source_lines[i].strip()
                shot_tgt = target_lines[i].strip()
                shot_prompt += f"\n{src_fullname}: " + shot_src + f"\n{tgt_fullname}: " + shot_tgt

        suffix = f"\n{tgt_fullname}:"
        prompt = prefix + shot_prompt0 + shot_prompt + f"\n{src_fullname}: " + ex[source_lang] + suffix
        return prompt

    def get_prompt(source_lang, target_lang, ex, shots_eval_dict={}, use_target_lang_prompt_eval=False):
        if len(shots_eval_dict) != 0:
            return get_prompt_few_shot(source_lang, target_lang, ex, shots_eval_dict)
        src_fullname = LANG_TABLE[source_lang]
        tgt_fullname = LANG_TABLE[target_lang]
        if use_target_lang_prompt_eval and target_lang != "en":
            prefix = PREFIX[target_lang]
            suffix = SUFFIX[target_lang]
        else:
            prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
            suffix = f"\n{tgt_fullname}:"
        prompt = prefix + ex[source_lang] + suffix
        return prompt
    
    def tokenize_function_test_single(ex, source_lang, target_lang, tokenizer, data_args, idx, source_path, target_path, shots_eval_dict):
    # 生成prompt
        prompt = get_prompt_few_shot(source_lang, target_lang, ex,idx, source_path, target_path, shots_eval_dict)
        print(prompt)
        # 对prompt进行tokenize
        tokenized_input = tokenizer(prompt, max_length=data_args.max_source_length, padding='max_length', truncation=True, add_special_tokens=True)

        # 如果需要prefix LM，创建prefix mask
        if data_args.use_prefix_lm:
            tokenized_input["prefix_mask"] = tokenized_input["attention_mask"]

        return tokenized_input

    if training_args.do_predict:
        trainer.args.prediction_loss_only = False
        lg_pairs = sorted(test_raw_data.keys())  # 确保每个设备按相同顺序打印

        for lg_pair in lg_pairs:
            test_dataset = test_raw_data[lg_pair]["test"]
            src_lang, tgt_lang = lg_pair.split("-")
            logger.info(f"*** Prediction for {lg_pair}***")
            with open(os.path.join(training_args.output_dir, f"test-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}"), "a", encoding="utf-8") as f:
                a=1
            for idx, examples in enumerate(test_dataset):
                feature_name = list(examples.keys())[0]
                source_lang, target_lang = feature_name.split("-")
                ex=examples[feature_name]
                # 对当前行进行tokenize
                print(feature_name)
                print(examples)
                print(ex)
                f=os.path.join(training_args.output_dir, f"test-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}")
                tokenized_input = tokenize_function_test_single(ex, src_lang, tgt_lang, tokenizer, data_args, idx, '/UNICOMFS/hitsz_mzhang_1/lc/ALMA/ALMA/doc/gf/TEST_1/test.zh', f,shots_eval_dict_1)

                # 使用模型进行推理
                preds = trainer.predict(
                    test_dataset=[tokenized_input], 
                    max_new_tokens=data_args.max_new_tokens, 
                    num_beams=data_args.num_beams, 
                    metric_key_prefix="test",
                    use_cache=True,
                )[0]

                # 处理推理结果
                if int(torch.cuda.current_device()) == 0:
                    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                    decoded_pred = tokenizer.decode(preds[0], skip_special_tokens=True).strip()
                n = min(6, idx + 3)
                # 打开文件，追加处理后的预测结果，然后关闭文件
                with open(os.path.join(training_args.output_dir, f"test-{src_lang}-{tgt_lang}{data_args.suffix_eval_file}"), "a", encoding="utf-8") as f:
                    suffix = get_key_suffix(tgt_lang, data_args)
                    cleaned_pred = clean_outputstring(decoded_pred, suffix, logger, n)
                    f.write(cleaned_pred + "\n")

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

