
# Implementation from UniDetox (https://github.com/EminLU/UniDetox/blob/main/unidetox/toxic_gpt2_finetune_and_distill.py)

import json

import tqdm
import datasets


targets_of_interest = {
    "unidetox": [
        'wom','trans','gendermin','bis','gay','gay.man','gay.wom',
        'mixed.race','ethnic.minority','indig','indig.wom','non.white','bla','bla.wom','bla.man',
        'asi','asi.wom','asi.east','asi.south','asi.chin','asi.pak','arab',
        'eastern.europe','russian','pol','hispanic','immig','asylum','ref','for',
        'jew','mus','mus.wom','other.religion'
    ]
}


def process_dghs_for_train(dataset_path: str, output_jsonl_file: str) -> None:
    dataset = datasets.Dataset.from_csv(dataset_path)

    def filter_func(example):
        return example['target'] in targets_of_interest["unidetox"] and example['label'] == 'hate'

    dghs_filtered = dataset.filter(filter_func)
    
    with open(output_jsonl_file, 'a') as f:
        for item in tqdm.tqdm(dghs_filtered):
            text = item['text']
            json_line = json.dumps({"messages": [{"role": "assistant", "content": text}]})
            f.write(json_line + '\n')


def process_dghs_for_detox(dataset_path: str, output_jsonl_file: str) -> None:
    dataset = datasets.Dataset.from_csv(dataset_path)
    
    def filter_func(example):
        return example['target'] in targets_of_interest["unidetox"] and example['label'] == 'hate'

    dghs_filtered = dataset.filter(filter_func)
    with open(output_jsonl_file, 'a') as f:
        for item in tqdm.tqdm(dghs_filtered):
            text = item['text']
            json_line = json.dumps({"text": text})
            f.write(json_line + '\n')

    
if __name__ == "__main__":
    dghs_path = "/path/to/DGHS/Dynamically Generated Hate Dataset v0.2.3.csv"
    dghs_out_path = "data/dghs_for_toxic_train.jsonl"
    process_dghs_for_train(dghs_path, dghs_out_path)
    
    dghs_out_path_for_detox = "data/dghs_to_detox.jsonl"
    process_dghs_for_detox(dghs_path, dghs_out_path_for_detox)