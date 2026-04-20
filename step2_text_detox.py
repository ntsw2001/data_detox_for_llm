
import os
import json
import argparse

import tqdm
import torch
import datasets
import omegaconf
import transformers
from typing import Callable, Dict, List, Any, Optional

from modules.utils import ModelType, save_args
from modules.diffs_funcs import DiffsFuncs


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text Detoxification")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def example_process(example: Dict[str, Any], model_type: ModelType, tokenizer: transformers.PreTrainedTokenizer, max_length: int, shots_num: int = 0) -> Dict[str, str]:
    system_prompt = "You are a helpful assistant. Your task is to transcribe directly or indirectly harmful content in the text, "\
                    "as well as other content that may harm the model, into legal and reasonable content that conforms to human values."
                    
    # shots
    detox_shots_list = ["\nHere is an example input: this is a terrible fact.\nOutput: <answer>this is a bad fact.</answer>\n"]
    detox_shots = "".join(detox_shots_list[:shots_num]) if shots_num > 0 else ""
    
    detox_prefix = "You should return the result between \'<answer>\' and \'</answer>\'."
    detox_tail =  "\nHere is the text that may contain harmful content:\n\n"
                   
    assistant_prompt = "Here is the transcription result:\n\n<answer>"
    
    if model_type in [ModelType.QWEN2]:
        input_prompt = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" \
                       + detox_prefix + detox_shots + detox_tail + example["text"] + "<|im_end|>\n<|im_start|>assistant\n" + assistant_prompt
    elif model_type in [ModelType.GPT2]:
        input_prompt = system_prompt + "\n\n" + detox_prefix + detox_shots + detox_tail + example["text"] + "\n\n" + assistant_prompt
        
    tokenized_inputs = tokenizer(
        input_prompt,
        add_special_tokens=False,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        pad_to_multiple_of=8
    )
    
    return {
        "raw_text": example["text"],
        "input_prompt": input_prompt,
        "input_ids": tokenized_inputs["input_ids"].squeeze(0),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze(0)
    }
    

def collate_fn(batch: list) -> Dict[str, torch.Tensor | List[str]]:
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    raw_texts = [item["raw_text"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "raw_texts": raw_texts
    }

        
class DetoxifyTokenProcessor(transformers.LogitsProcessor):
    def __init__(
        self, 
        toxic_model: transformers.PreTrainedModel, 
        vocab_size: int, 
        max_gen_repeat_times: int,
        max_cache_len: int,
        detox_mode_name: str, 
        detox_args: Dict[str, Any], 
        diffs_fn: Optional[Callable], 
        diffs_fn_args: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.toxic_model = toxic_model
        self.vocab_size = vocab_size
        self.max_gen_repeat_times = max_gen_repeat_times
        self.detox_mode_name = detox_mode_name
        self.detox_args = detox_args
        self.diffs_fn = diffs_fn
        self.diffs_fn_args = diffs_fn_args
        
        # toxic model kv cache
        self._max_cache_len = max_cache_len
        self._first_step = True
        self._pos = 0
        self._toxic_cache = transformers.StaticCache(
            config=self.toxic_model.config,
            max_cache_len=self._max_cache_len,
            offloading=False,
            offload_only_non_sliding=False
        )

    
    def reset(self) -> None:
        """
        reset static cache of toxic_model before each base model generation.
        """
        del self._toxic_cache
        self._toxic_cache = transformers.StaticCache(
            config=self.toxic_model.config,
            max_cache_len=self._max_cache_len,
            offloading=False,
            offload_only_non_sliding=False
        )
        self._first_step = True
        self._pos = 0
    

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.detox_mode_name == "prompt_only":
            return scores
        
        with torch.inference_mode():
            if self._first_step:
                self._pos = input_ids.size(1)
                cache_position = torch.arange(self._pos, device=input_ids.device)
                toxic_outputs = self.toxic_model(
                    input_ids=input_ids, 
                    cache_position=cache_position, 
                    past_key_values=self._toxic_cache, 
                    use_cache=True
                )
                self._first_step = False
            else:
                cache_position = torch.tensor([self._pos], device=input_ids.device)
                toxic_outputs = self.toxic_model(
                    input_ids[..., -1:],
                    cache_position=cache_position,
                    past_key_values=self._toxic_cache,
                    use_cache=True
                )
                self._pos += 1
            
        toxic_logits: torch.Tensor = toxic_outputs.logits[..., -1, :self.vocab_size].to(torch.float32)
        tail_scores = scores[:, self.vocab_size:]
        scores = scores[:, :self.vocab_size]
        
        if self.detox_mode_name == "vanilla_cd":
            if self.diffs_fn is not None:
                cd_logits = torch.empty_like(scores)
                for batch_idx in range(scores.shape[0]):
                    diff_score = self.diffs_fn(scores[batch_idx].unsqueeze(0), toxic_logits[batch_idx].unsqueeze(0), **self.diffs_fn_args)
                    cutoff_score = torch.log(diff_score) + scores[batch_idx].max(dim=-1, keepdim=True).values
                    # http://arxiv.org/abs/2309.09117
                    # logits diffs
                    diffs = (1 + self.detox_args.beta1) * scores[batch_idx] - self.detox_args.beta2 * toxic_logits[batch_idx]
                    cd_logits[batch_idx] = diffs.masked_fill(scores[batch_idx] < cutoff_score, -1e8)
            else:
                # http://arxiv.org/abs/2309.09117
                # beta1 = 0, beta2 = 1, is https://aclanthology.org/2023.acl-long.687
                cutoff_score = torch.log(torch.tensor(self.detox_args.alpha)) + scores.max(dim=-1, keepdim=True).values
                diffs = (1 + self.detox_args.beta1) * scores - self.detox_args.beta2 * toxic_logits
                cd_logits = diffs.masked_fill(scores < cutoff_score, -1e8)
            # concat scores and tail_scores
            cd_logits = torch.cat([cd_logits, tail_scores], dim=-1)
            return cd_logits
        
        elif self.detox_mode_name == "socd":
            V = scores.size(-1)
            for batch_idx in range(scores.shape[0]):
                raw_diff = self.diffs_fn(
                    scores[batch_idx].unsqueeze(0),
                    toxic_logits[batch_idx].unsqueeze(0),
                    **self.diffs_fn_args
                )
                if not torch.is_tensor(raw_diff):
                    raw_diff = torch.tensor(raw_diff, dtype=scores.dtype, device=scores.device)
                raw_diff = raw_diff.to(dtype=torch.float32, device=scores.device)

                # handle edge case
                if not torch.isfinite(raw_diff):
                    alpha = torch.tensor(0.0, dtype=torch.float32, device=scores.device)
                else:
                    t = torch.log1p(torch.clamp(raw_diff, min=0.0))
                    alpha = (t / (1.0 + t)).clamp(0.0, 1.0)

                # adaptive top-k：alpha * V, min=1
                if self.detox_args.k_setting == "dynamic":
                    k = int(torch.clamp((alpha * V).floor(), min=self.detox_args.top_k_num, max=V).item())
                else:
                    k = self.detox_args.top_k_num

                # log_prob diffs
                lp_base = torch.log_softmax(scores[batch_idx], dim=-1)
                lp_toxic = torch.log_softmax(toxic_logits[batch_idx], dim=-1)
                diff = lp_toxic - lp_base

                # keep top-k positive diffs, others set to -inf
                diff_pos = torch.where(
                    diff > 0,
                    diff,
                    torch.tensor(float('-inf'), device=diff.device, dtype=diff.dtype)
                )

                # fetch top-k indices and values among positive diffs
                topk_vals, topk_idx = torch.topk(diff_pos, k, dim=-1)
                valid_mask = torch.isfinite(topk_vals)
                if valid_mask.any():
                    sel_idx = topk_idx[valid_mask]

                    # base - alpha * toxic
                    scores[batch_idx, sel_idx] = scores[batch_idx, sel_idx] - alpha * toxic_logits[batch_idx, sel_idx]

            scores = torch.cat([scores, tail_scores], dim=-1)
            return scores
        
        else:
            raise ValueError(f"Unsupported detox mode: {self.detox_mode_name}.")
            
            
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.recompile_limit = 2**31
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Current device: {device}")
    
    init_args = arg_parser()
    args = omegaconf.OmegaConf.load(init_args.config)
    transformers.set_seed(args.general.seed)
    
    # tokenizer
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_path)
    if args.model.model_type in [ModelType.GPT2]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # dataset
    toxic_text_dataset: datasets.Dataset = datasets.load_dataset("json", data_files=args.data.input_file, split="train")
    tokenized_dataset = toxic_text_dataset.map(
        example_process,
        fn_kwargs={"model_type": ModelType(args.model.model_type), "tokenizer": tokenizer, "max_length": args.gen.max_prompt_length, "shots_num": args.detox.shots_num},
        remove_columns=toxic_text_dataset.column_names,
        desc="Tokenizing dataset"
    )
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=args.gen.batch_size,
        collate_fn=lambda batch: collate_fn(batch),
        shuffle=False,
        pin_memory=True,
        num_workers=args.data.data_workers,
        persistent_workers=True, 
        prefetch_factor=2
    )

    # model
    toxic_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model.toxic_model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
        low_cpu_mem_usage=True
    )
    toxic_model.eval()
    toxic_model.to(device)
    toxic_model = torch.compile(toxic_model, mode="default", fullgraph=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
        low_cpu_mem_usage=True
    )
    model.eval()
    model.to(device)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    
    # logits processor
    # Attention: we provide the implementation of DetoxifyTokenProcessor, which is actually worked just after RepetitionPenaltyLogitsProcessor
    detox_processor = DetoxifyTokenProcessor(
        toxic_model=toxic_model,
        vocab_size=tokenizer.vocab_size,
        max_gen_repeat_times=args.gen.max_gen_repeat_times,
        max_cache_len=args.gen.max_new_tokens + args.gen.max_prompt_length + 32,
        detox_mode_name=args.gen.detox_mode_name,
        detox_args=args.detox[args.gen.detox_mode_name],
        diffs_fn=DiffsFuncs[args.detox.diffs_fn.diffs_fn_name],
        diffs_fn_args=args.detox.diffs_fn.diffs_fn_args
    )
    logits_processor = transformers.LogitsProcessorList([detox_processor])
    print(f">>> Current <diffs_fn_name>: {args.gen.detox_mode_name}")
    print(f">>> Current <diffs_fn_name>: {args.detox.diffs_fn.diffs_fn_name}")

    # gen config
    generation_config = transformers.GenerationConfig.from_pretrained(args.model.model_path)
    generation_config.max_new_tokens = args.gen.max_new_tokens
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.do_sample = args.gen.do_sample
    generation_config.num_return_sequences = args.gen.max_gen_repeat_times
    generation_config.use_cache = True
    generation_config.cache_implementation = "static"
    generation_config.top_p = 1.0
    
    # output config
    if args.general.is_resumed:
        output_dir = args.data.output_dir + "/" + args.general.resume_dir
    else:
        output_dir = save_args(args, args.data.output_dir)
    output_file = os.path.join(output_dir, f"dghs_{args.model.model_type}_{args.model.model_size_info}_{args.gen.detox_mode_name}_{args.detox.diffs_fn.diffs_fn_name}.jsonl")
    
    # resuming config
    if args.general.is_resumed and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existed_lines = f.readlines()
        batch_existed = len(existed_lines) // args.gen.batch_size
        print(f">>> Resuming from existing file: {output_file}")
        print(f">>> Existing batches with batch_size={args.gen.batch_size}: {batch_existed}")
    else:
        batch_existed = 0
    
    # generation
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Detoxifying text"):
        # resuming
        if idx < batch_existed: continue
        
        # inference
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        bs = input_ids.size(0)
        detoxified_texts_list = [[] for _ in range(bs)]
        
        for temperature in args.gen.temperature_list:
            generation_config.temperature = temperature
            # reset static cache of toxic_model before each generation to avoid potential cross-contamination among generations
            detox_processor.reset()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    logits_processor=logits_processor
                )

            prompt_lens = input_ids.size(1)
            for i in range(bs):
                start = outputs.size(1) - prompt_lens
                seq_block = outputs[i * args.gen.max_gen_repeat_times : (i + 1) * args.gen.max_gen_repeat_times, -start - 3:]
                texts = tokenizer.batch_decode(seq_block, skip_special_tokens=True)
                for t_idx, text in enumerate(texts, start=1):
                    detoxified_texts_list[i].append(
                        {"temperature": temperature, "current_time": t_idx, "text": text}
                    )
        
        # Save results
        for i, detoxified_text_list in enumerate(detoxified_texts_list):
            gen_result = {
                "raw_text": batch["raw_texts"][i],
                "detoxified_texts": [text_dict["text"] for text_dict in detoxified_text_list],
                "output": detoxified_text_list
            }
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(gen_result, ensure_ascii=False) + "\n")
                
    print("Done.")
