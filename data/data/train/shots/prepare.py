import argparse
import json
import random

def read_lines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def get_translation_pairs(de_sentences, en_sentences, index, use_random):
    # Choose a random number of pairs between 1 and 5
    num_pairs = random.randint(1, 5)
    pairs = []
    
    if use_random or index < num_pairs:
        chosen_indices = random.sample(range(len(de_sentences)), num_pairs)
    else:
        chosen_indices = range(index - num_pairs, index)

    for i in chosen_indices:
        pairs.append({
            "source": de_sentences[i],
            "target": en_sentences[i]
        })

    return pairs

def create_json_output(de_sentences, en_sentences):
    output_data = {}
    for index, de_sentence in enumerate(de_sentences):
        use_random = random.random() > 0.5
        translation_pairs = get_translation_pairs(de_sentences, en_sentences, index, use_random)
        output_data[de_sentence] = translation_pairs
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Create translation pairs JSON from DE and EN text files.')
    parser.add_argument('-s', '--source', required=True, type=str, help='Path to the source German text file (de.txt)')
    parser.add_argument('-t', '--target', required=True, type=str, help='Path to the target English text file (en.txt)')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to the output JSON file (output.json)')
    
    args = parser.parse_args()

    de_sentences = read_lines(args.source)
    en_sentences = read_lines(args.target)
    
    if len(de_sentences) != len(en_sentences):
        raise ValueError("The number of sentences in de.txt and en.txt must be the same.")
    
    output_data = create_json_output(de_sentences, en_sentences)
    
    with open(args.output, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

#python generate_json.py -s train.de -t train.en -o shots.de-en.json