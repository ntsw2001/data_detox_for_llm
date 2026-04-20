
import os
import json
import time
import enum

import torch
import omegaconf
import accelerate
import transformers
from typing import Tuple, Optional


class ModelType(str, enum.Enum):
    GPT2 = "gpt2"
    QWEN2 = "qwen2"
    

def get_model_and_tokenizer_offline(
    model_type: str, 
    model_name: str, 
    tokenizer_name: str, 
    num_classes: int, 
    state_dict: torch.nn.Module.state_dict, 
    huggingface_config_path: Optional[str] = None
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    model_class = getattr(transformers, model_name)
    config = model_class.config_class.from_pretrained(huggingface_config_path or model_type, num_labels=num_classes)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,
    )
    return model, tokenizer


def save_args(args_dict: omegaconf.OmegaConf, arg_output_dir: str, accelerator: accelerate.Accelerator = None) -> str:
    if accelerator is None:
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        output_dir = os.path.join(arg_output_dir, time_str)
        os.makedirs(output_dir, exist_ok=True)
        current_args = omegaconf.OmegaConf.to_container(args_dict, resolve=True)
        with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(current_args, f, ensure_ascii=False, indent=4)
        return output_dir
    
    else:
        if accelerator.is_main_process:
            timestamp_float = time.time()
            timestamp_tensor = torch.tensor([timestamp_float], dtype=torch.float32, device=accelerator.device)
        else:
            timestamp_tensor = torch.empty(1, dtype=torch.float32, device=accelerator.device)
        timestamp_tensor = accelerate.utils.broadcast(timestamp_tensor)
        shared_timestamp = timestamp_tensor.item()
        shared_timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(shared_timestamp))
        
        output_dir = os.path.join(arg_output_dir, shared_timestamp_str)
        
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            current_args = omegaconf.OmegaConf.to_container(args_dict, resolve=True)
            with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(current_args, f, ensure_ascii=False, indent=4)
            
        return output_dir