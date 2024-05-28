import json
import jieba
import math
from nltk.tokenize import word_tokenize

def preprocess(text, language='zh'):
    """根据语言进行文本预处理和分词。"""
    if language == 'zh':
        return list(jieba.cut(text))
    else:
        return word_tokenize(text.lower())

def extract_ngrams(text, n=2, language='zh'):
    """提取文本的n-gram列表。"""
    words = preprocess(text, language)
    ngrams = [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return ngrams

def calculate_overlap_score(weighted_ngrams, example_ngrams):
    """计算重叠得分。"""
    matched_ngrams_count = sum(1 for ngram in example_ngrams if ngram in weighted_ngrams)
    total_ngrams_count = len(example_ngrams)
    return matched_ngrams_count / total_ngrams_count if total_ngrams_count > 0 else 0

def select_bm25_examples_iteratively(test_source, bm25_examples, language='zh', lambda_factor=0.5):
    new_examples = []
    source_ngrams_sets = [set(extract_ngrams(test_source, n, language)) for n in range(1, 5)]
    ngram_weights = [{ngram: 1.0 for ngram in ngrams_set} for ngrams_set in source_ngrams_sets]

    while bm25_examples:
        scores = []
        for example in bm25_examples:
            example_scores = []
            for n, source_ngrams in enumerate(source_ngrams_sets, start=1):
                example_ngrams = extract_ngrams(example['source'], n, language)
                score = calculate_overlap_score(ngram_weights[n-1], example_ngrams)
                example_scores.append(math.log(score + 0.000001))  # 避免log(0)
            final_score = math.exp(sum(example_scores) / len(example_scores))
            scores.append((example, final_score))

        # 选择分数最高的示例并从bm25_examples中移除
        selected_example, _ = max(scores, key=lambda x: x[1])
        bm25_examples.remove(selected_example)
        new_examples.append(selected_example)

        # 更新权重
        for n, ngram_weight in enumerate(ngram_weights, start=1):
            selected_ngrams = extract_ngrams(selected_example['source'], n, language)
            for ngram in selected_ngrams:
                if ngram in ngram_weight:
                    ngram_weight[ngram] *= lambda_factor

    return new_examples

# 读取 JSON 文件
def process_bm25_file(input_file, output_file, language='zh'):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_data = {}
    for key, examples in data.items():
        # 假设 key 是测试源文本
        test_source = key
        bm25_examples = examples
        selected_examples = select_bm25_examples_iteratively(test_source, bm25_examples, language)
        processed_data[key] = selected_examples

    # 将处理后的数据保存为新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)

# 示例使用
input_file = '/data/lc/ALMA/ALMA/human_written_data_domain_subtitles/select_shots/shots.de-en.json'
output_file = '/data/lc/ALMA/ALMA/human_written_data_domain_subtitles/r-bm25_shots/shots.de-en.json'
process_bm25_file(input_file, output_file, 'de')  # 假设处理的是德文数据
