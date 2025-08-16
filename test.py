# import os
# import gc
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer
# from activation_aware_pruning import ActivationAwarePruning
# from config import PruningConfig, ActivationAnalysisConfig, EvaluateConfig
# from lib.model_utils import get_llm

# # =========================================================================
# # 步驟 1：定義 Hugging Face 模型 ID 和模型緩存的根目錄
# HF_MODEL_ID = "meta-llama/Llama-2-7b-hf"
# LOCAL_CACHE_DIR = "llm_weights"

# # =========================================================================
# # 步驟 2：載入模型以便分析 activation（此步會下載權重）
# print(f"嘗試載入模型: {HF_MODEL_ID}，緩存目錄: {LOCAL_CACHE_DIR}")
# model = get_llm(HF_MODEL_ID, cache_dir=LOCAL_CACHE_DIR)
# tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=False)
# print(f"模型 {HF_MODEL_ID} 已載入或存在於 {LOCAL_CACHE_DIR}")

# # 設定 tokenizer 的 pad_token
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # =========================================================================
# # 步驟 3：準備 C4 資料（採集 128 個至少 2048 tokens 的樣本）
# def get_filtered_texts_streaming(dataset_name, subset, split, min_tokens=2048, max_samples=128):
#     print(f"Streaming 載入 {dataset_name}/{subset}，分割：{split} ...")
#     ds_stream = load_dataset(dataset_name, subset, split=split, streaming=True)
#     filtered = []
#     for ex in ds_stream:
#         tokens = tokenizer(ex["text"], truncation=False)["input_ids"]
#         if len(tokens) >= min_tokens:
#             filtered.append(ex["text"])
#         if len(filtered) >= max_samples:
#             break
#     print(f"取得 {len(filtered)} 筆樣本（每筆至少 {min_tokens} tokens）")
#     return filtered

# texts = get_filtered_texts_streaming("allenai/c4", "en.noblocklist", "train", min_tokens=2048, max_samples=128)

# # =========================================================================
# # 步驟 4：分析 activation 最豐富的樣本並儲存
# pruner = ActivationAwarePruning(LOCAL_CACHE_DIR)
# analysis_config = ActivationAnalysisConfig(
#     model=model,
#     tokenizer=tokenizer,
#     dataset_texts=texts,
#     top_k=128,
#     layer_index=0,
#     device="mps",
#     save_path="./topk_c4_gradient.jsonl"
# )
# print("分析啟用最豐富的樣本中...")
# stats = pruner.analyze_activation(analysis_config)

# # 清除記憶體
# del model
# gc.collect()
# torch.cuda.empty_cache()

# # =========================================================================
# # 步驟 5：進行剪枝
# pruning_config = PruningConfig(model=HF_MODEL_ID)
# pruner.prune_wanda(pruning_config)
# pruner.set_model_path("out/hybrid_coverage_30percent")

# eval_config = EvaluateConfig(
#     ntrain=5,
#     lora_path=None,
#     data_dir="data",
#     save_dir="output",
#     engine=["hybrid_coverage_30percent"],
#     eval_mode="full",
#     custom_subjects=None
# )
# print("剪枝後進行評估中...")
# pruner.evaluate(eval_config)

'''
from activation_aware_pruning import ActivationAwarePruning
from config import PruningConfig, EvaluateConfig, HybridDatasetConfig, DatasetConfig

hybrid_config = HybridDatasetConfig(
        datasets=[
            DatasetConfig("wikitext2", weight=0.5, nsamples=50),
            DatasetConfig("c4", weight=0.5, nsamples=40),
            # 如果有自定義的JSON檔案，可以這樣配置：
            # DatasetConfig("json", weight=0.1, file_path="path/to/custom.jsonl", nsamples=13)
        ],
        total_nsamples=128,
        mixing_strategy="balanced",  # 平衡混合策略
        save_path="out/hybrid_dataset_wanda.json",
        seed=42
    )

aacp = ActivationAwarePruning("/Users/timmy/Code/AACP/llm_weights/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9")
aacp.create_hybrid_dataset(hybrid_config)
'''

from activation_aware_pruning import ActivationAwarePruning as aacp
from config import PruningConfig, EvaluateConfig

model_path = "/Users/timmy/Code/AACP/model/models--meta-llama--Llama-2-7b-hf"

aacp = aacp(model_path)

pruning_config = PruningConfig(
    model="meta-llama/Llama-2-7b-hf",
    dataset="/home/timmy930511/AACP/dataset/hybrid_dataset_wanda.json",
    save_model = "out/wanda_c4_wikitext2"
)

evaluate_config = EvaluateConfig(
    engine=["wanda_c4_wikitext2"]
)

aacp.prune_wanda(pruning_config)
aacp.evaluate(evaluate_config)
