# data_detox_for_llm

Official code for the paper **Detoxification for LLM: From Dataset Itself** (ACL 2026 Main, Poster). The code is based on transformers.



## Environment Requirements

- CUDA 12.1
- Python 3.12.13

```bash
git lfs install
git clone https://github.com/ntsw2001/data_detox_for_llm.git
pip install -r requirements.txt
```



## Data and Model Preparation

Datasets used in this work:

| **Name**                                                  | **URL**                                                      | **Abbreviation in this work** |
| --------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------- |
| LennardZuendorf/Dynamically-Generated-Hate-Speech-Dataset | https://huggingface.co/datasets/LennardZuendorf/Dynamically-Generated-Hate-Speech-Dataset | DGHS                          |
| toxigen/toxigen-data                                      | https://huggingface.co/datasets/toxigen/toxigen-data         |                               |
| cais/mmlu                                                 | https://huggingface.co/datasets/cais/mmlu                    |                               |

Main models used in this work:

| **Name**                      | **URL**                                                      | **Remarks**                                                  |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| openai-community/gpt2-xl      | https://huggingface.co/openai-community/gpt2-xl              |                                                              |
| Qwen/Qwen2.5-0.5B-Instruct    | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct            |                                                              |
| Qwen/Qwen2.5-3B-Instruct      | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct              |                                                              |
| Detoxify/original             | [https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_original-c1212f89.ckpt](https://www.google.com/search?q=https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_original-c1212f89.ckpt) | When reading offline, the `bert-base-uncased` structure needs to be loaded first. |
| google-bert/bert-base-uncased | https://huggingface.co/google-bert/bert-base-uncased         |                                                              |
| meta-llama/Llama-2-7b-hf      | https://huggingface.co/meta-llama/Llama-2-7b-hf              |                                                              |
| facebook/opt-6.7b             | https://huggingface.co/facebook/opt-6.7b                     |                                                              |
| tiiuae/falcon-7b              | https://huggingface.co/tiiuae/falcon-7b                      |                                                              |
| Qwen/Qwen3-Embedding-0.6B     | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B             |                                                              |



## Prerequisites

**Data Preprocessing**

```bash
python step0_data_prepare.py
```

**Fine-tuning the small toxic model**

```bash
bash scripts/step1_finetune_toxic.sh
```



## Detoxification

**SoCD**

```bash
bash scripts/step2_detoxify_texts.sh
```

**Fusion Ranking**

```bash
bash scripts/step3_toxic_rank.sh
```



## Evaluation

**Fine-tuning the model with detoxified texts**

```bash
bash scripts/step4-1_toxic_eval_sft.sh
```

**Evaluating model toxicity**

```bash
bash scripts/step4-2_toxic_eval.sh
```



## Additional Notes

We provide the detoxified text files that achieved the best detoxification results for GPT2-XL as shown in table 1 of the paper. This includes the intermediate files obtained from the detoxification step (step 2) (`data/step2/dghs_qwen2_3-0.5_socd_wsd.jsonl`) and the intermediate files from Fusion Ranking (step 3) (`data/step3/*.jsonl`). To reproduce this work, you can directly use `data/step3/swift_dghs_ranked_dim-1024_cosine_similarity_from-_dghs_qwen2_3-0.5_top_k_cd_wsd_.jsonl` to fine-tune GPT2-XL and evaluate it.



If you find our work helpful, please cite our paper (TBD)
