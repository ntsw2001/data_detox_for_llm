import os
import re
import json
import argparse

import tqdm
import torch
import datasets
import omegaconf
import transformers
import sentence_transformers
from detoxify import Detoxify, detoxify
from typing import Optional, Tuple, Dict, Any

from modules.utils import get_model_and_tokenizer_offline, save_args


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text Detoxification")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def example_extractor(example: Dict[str, Any]) -> Dict[str, Any]:
    detoxified_texts = example["detoxified_texts"]
    extracted_answers = []
    for text in detoxified_texts:
        match = re.match(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            extracted_answers.append(match.group(1))
        else:
            extracted_answers.append(text.split("<answer>")[0].strip())
    return {"raw_text": example["raw_text"], "extracted_answers": list(set(extracted_answers))}


def example_toxicity_eval(example: Dict[str, Any], detoxify_model: Detoxify) -> Dict[str, Any]:
    extracted_answers = example["extracted_answers"]
    
    origin_toxicity_scores: Dict[str, float] = detoxify_model.predict(example["raw_text"])
    fused_origin_toxicity_score = origin_toxicity_scores["toxicity"] * (1 + origin_toxicity_scores["severe_toxicity"])
    
    # evaluate toxicity scores
    score_dicts = []
    ans_scores = detoxify_model.predict(extracted_answers)

    for ans_idx, ans_txt in enumerate(extracted_answers):
        # {'toxicity': 0.123, 'severe_toxicity': 0.045, 'obscene': 0.067, 'threat': 0.001, 'insult': 0.034, 'identity_attack': 0.012}
        fused_toxicity_score = ans_scores["toxicity"][ans_idx]
        # fused_toxicity_score = ans_scores["toxicity"][ans_idx] * (1 + ans_scores["severe_toxicity"][ans_idx])
        score_dicts.append({
            "answer": ans_txt,
            "fused_toxicity_score": fused_toxicity_score,
            "toxicity_scores": {detox_key: detox_value[ans_idx] for detox_key, detox_value in ans_scores.items()},
        })

    return {
        "toxicity_scores": score_dicts,
        "fused_origin_toxicity_score": fused_origin_toxicity_score,
        "origin_toxicity_scores": origin_toxicity_scores
    }


def example_semantic_rank_cos(example: Dict[str, Any], embedding_model: sentence_transformers.SentenceTransformer) -> Dict[str, Any]:
    extracted_answers = example["extracted_answers"]
    embeddings: torch.Tensor = embedding_model.encode(extracted_answers, convert_to_tensor=True)
    raw_text_embedding = embedding_model.encode(example["raw_text"], convert_to_tensor=True)
    
    score_dicts = []
    for answer_text, answer_emb in zip(extracted_answers, embeddings):
        cosine_similarity = torch.nn.functional.cosine_similarity(raw_text_embedding, answer_emb, dim=-1).item()
        score_dicts.append({
            "answer": answer_text,
            "cosine_similarity": cosine_similarity
        })
    return {"semantic_scores": score_dicts}


def example_get_final_detoxified_text(example: Dict[str, Any]) -> Dict[str, Any]:
    toxicity_scores = example["toxicity_scores"]
    semantic_scores = example["semantic_scores"]
    
    final_scores = []
    for tox, sem in zip(toxicity_scores, semantic_scores):
        score = -tox["fused_toxicity_score"] + sem["cosine_similarity"]
        # score = -tox["fused_toxicity_score"] # lambda = 1
        # score = sem["cosine_similarity"] # lambda = 0
        final_scores.append(score)

    best_idx = int(torch.tensor(final_scores).argmax().item())
    final_text = toxicity_scores[best_idx]["answer"]
    final_text_semantic_score = semantic_scores[best_idx]["cosine_similarity"]
    final_text_fused_toxicity_score: Dict[str, float] = toxicity_scores[best_idx]["fused_toxicity_score"]
    final_text_all_toxicity_scores = toxicity_scores[best_idx]["toxicity_scores"]
    
    return {
        "final_detoxified_text": final_text,
        "fused_toxicity_score": final_text_fused_toxicity_score,
        "all_toxicity_scores": final_text_all_toxicity_scores,
        "semantic_score": final_text_semantic_score
    }


if __name__ == "__main__":
    init_args = arg_parser()
    args = omegaconf.OmegaConf.load(init_args.config)
    transformers.set_seed(args.general.seed)
    detoxify.get_model_and_tokenizer = get_model_and_tokenizer_offline
    
    # data
    print(args.data.input_file)
    detoxic_text_dataset: datasets.Dataset = datasets.load_dataset("json", data_files=args.data.input_file, split="train")
    detoxic_text_dataset = detoxic_text_dataset.map(
        example_extractor,
        remove_columns=detoxic_text_dataset.column_names,
        desc="Extracting answers"
    )
    
    # Detoxify eval and rank
    detoxify_model = Detoxify(
        model_type=args.detoxify.model_type,
        checkpoint=args.detoxify.ckpt_path,
        huggingface_config_path=args.detoxify.huggingface_config_path,
        device=f"cuda:0"
    )
    detoxic_text_dataset = detoxic_text_dataset.map(
        example_toxicity_eval,
        fn_kwargs={"detoxify_model": detoxify_model},
        desc="Eval toxicity scores"
    )
    
    # semantic rank
    embedding_model = sentence_transformers.SentenceTransformer(
        args.embeddings.embedding_model_path,
        device=f"cuda:0",
        local_files_only=True,
        model_kwargs={"dtype": torch.bfloat16}
    )
    args.embeddings.dim_num = embedding_model.get_sentence_embedding_dimension()
    if args.rank_mode.rank_mode_name == "cosine_similarity":
        detoxic_text_dataset = detoxic_text_dataset.map(
            example_semantic_rank_cos,
            fn_kwargs={"embedding_model": embedding_model},
            desc="Eval semantic similarity scores"
        )
    
    # get final detoxified text
    detoxic_text_dataset = detoxic_text_dataset.map(
        example_get_final_detoxified_text,
        desc="Get final detoxified text"
    )
    
    # direct toxic score
    direct_toxic_score = {
        "after": {
            "avg_fused_toxic_score": sum(example["fused_toxicity_score"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_toxic_score": sum(example["all_toxicity_scores"]["toxicity"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_severe_toxicity": sum(example["all_toxicity_scores"]["severe_toxicity"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_obscene": sum(example["all_toxicity_scores"]["obscene"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_threat": sum(example["all_toxicity_scores"]["threat"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_insult": sum(example["all_toxicity_scores"]["insult"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_identity_attack": sum(example["all_toxicity_scores"]["identity_attack"] for example in detoxic_text_dataset) / len(detoxic_text_dataset)
        },
        "before": {
            "avg_fused_toxic_score": sum(example["fused_origin_toxicity_score"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_toxic_score": sum(example["origin_toxicity_scores"]["toxicity"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_severe_toxicity": sum(example["origin_toxicity_scores"]["severe_toxicity"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_obscene": sum(example["origin_toxicity_scores"]["obscene"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_threat": sum(example["origin_toxicity_scores"]["threat"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_insult": sum(example["origin_toxicity_scores"]["insult"] for example in detoxic_text_dataset) / len(detoxic_text_dataset),
            "avg_identity_attack": sum(example["origin_toxicity_scores"]["identity_attack"] for example in detoxic_text_dataset) / len(detoxic_text_dataset)
        }
    }
    direct_toxic_score["delta"] = {
        key: direct_toxic_score["after"][key] - direct_toxic_score["before"][key]
        for key in direct_toxic_score["after"]
    }

    # save
    output_dir = save_args(args, args.data.output_dir)
    input_file_name = os.path.splitext(os.path.basename(args.data.input_file))[0]
    output_file_name = f"{args.data.data_name}_ranked_dim-{args.embeddings.dim_num}_{args.rank_mode.rank_mode_name}_from-<{input_file_name}>.jsonl"
    output_file_path = os.path.join(output_dir, output_file_name)
    full_output_file_path = os.path.join(output_dir, "full_" + output_file_name)
    swift_output_file_path = os.path.join(output_dir, "swift_" + output_file_name)
    
    with open(output_file_path, "a") as f:
        for example in tqdm.tqdm(detoxic_text_dataset, desc="Saving results"):
            final_text = {"raw_text": example["raw_text"], "detoxified_texts": example["final_detoxified_text"]}
            f.write(json.dumps(final_text, ensure_ascii=False) + "\n")
    with open(full_output_file_path, "a") as f:
        for example in tqdm.tqdm(detoxic_text_dataset, desc="Saving full results"):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
        f.write(json.dumps(direct_toxic_score, ensure_ascii=False) + "\n")
    with open(swift_output_file_path, "a") as f:
        for example in tqdm.tqdm(detoxic_text_dataset, desc="Saving swift results"):
            f.write(json.dumps({"messages": [{"role": "assistant", "content": example["final_detoxified_text"]}]}, ensure_ascii=False) + "\n")
            
    print("Done.")