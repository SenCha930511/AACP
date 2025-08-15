import json
import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from config.hybrid_dataset_config import HybridDatasetConfig, DatasetConfig
from lib.data import get_loaders
from transformers import AutoTokenizer
import os

def set_seed(seed: int):
    """設定隨機種子"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def fix_model_path(model_path: str) -> str:
    """修復模型路徑，處理snapshot等特殊情況"""
    
    # 如果路徑包含snapshots，嘗試找到正確的模型目錄
    if "snapshots" in model_path:
        base_path = model_path.split("/snapshots/")[0]
        
        # 檢查基礎路徑是否包含必要的檔案
        if os.path.exists(os.path.join(base_path, "tokenizer.json")):
            print(f"找到正確的模型路徑: {base_path}")
            return base_path
        elif os.path.exists(os.path.join(base_path, "tokenizer.model")):
            print(f"找到正確的模型路徑: {base_path}")
            return base_path
        elif os.path.exists(os.path.join(base_path, "tokenizer_config.json")):
            print(f"找到正確的模型路徑: {base_path}")
            return base_path
    
    # 檢查原始路徑是否包含必要的檔案
    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        return model_path
    elif os.path.exists(os.path.join(model_path, "tokenizer.model")):
        return model_path
    elif os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        return model_path
    
    # 如果都找不到，返回原始路徑（讓transformers處理錯誤）
    return model_path

def ensure_directory_exists(path: str, description: str = "目錄"):
    """確保目錄存在，如果不存在則創建"""
    if path and not os.path.exists(path):
        try:
            print(f"創建{description}: {path}")
            os.makedirs(path, exist_ok=True)
            print(f"✅ 成功創建{description}: {path}")
        except Exception as e:
            print(f"❌ 創建{description}失敗: {path}, 錯誤: {e}")
            raise
    elif path and os.path.exists(path):
        print(f"✅ {description}已存在: {path}")
    return path

def load_tokenizer_from_model(model_path: str):
    """從模型路徑自動載入tokenizer"""
    try:
        print(f"從模型路徑載入tokenizer: {model_path}")
        
        # 處理可能的snapshot路徑問題
        if "snapshots" in model_path:
            # 如果是snapshot路徑，嘗試使用父目錄
            base_path = model_path.split("/snapshots/")[0]
            print(f"檢測到snapshot路徑，嘗試使用基礎路徑: {base_path}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
                print(f"從基礎路徑成功載入tokenizer")
            except Exception as e:
                print(f"從基礎路徑載入失敗: {e}")
                # 如果基礎路徑也失敗，嘗試原始路徑
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        else:
            # 直接使用提供的路徑
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        # 檢查並設定pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"設定pad_token為eos_token: {tokenizer.eos_token}")
            else:
                tokenizer.pad_token = tokenizer.eos_token = "<pad>"
                print("設定pad_token和eos_token為: <pad>")
        
        print(f"成功載入tokenizer，詞彙大小: {tokenizer.vocab_size}")
        return tokenizer
        
    except Exception as e:
        print(f"載入tokenizer時發生錯誤: {e}")
        print(f"請檢查模型路徑: {model_path}")
        print("建議:")
        print("1. 確保路徑指向包含tokenizer.json的目錄")
        print("2. 如果是snapshot路徑，嘗試使用父目錄")
        print("3. 檢查目錄是否包含必要的tokenizer檔案")
        raise

def calculate_sample_distribution(config: HybridDatasetConfig) -> Dict[str, int]:
    """計算每個資料集的樣本數量分配"""
    total_weight = sum(ds.weight for ds in config.datasets)
    dataset_samples = {}
    
    for ds in config.datasets:
        if ds.nsamples is not None:
            # 如果明確指定了樣本數量
            dataset_samples[ds.name] = ds.nsamples
        else:
            # 根據權重分配樣本數量
            ds_nsamples = int((ds.weight / total_weight) * config.total_nsamples)
            # 確保每個資料集至少有最小樣本數
            ds_nsamples = max(config.min_samples_per_dataset, ds_nsamples)
            dataset_samples[ds.name] = ds_nsamples
    
    # 調整總數以符合要求
    total_allocated = sum(dataset_samples.values())
    if total_allocated != config.total_nsamples:
        # 按比例調整
        scale_factor = config.total_nsamples / total_allocated
        for name in dataset_samples:
            dataset_samples[name] = max(1, int(dataset_samples[name] * scale_factor))
    
    return dataset_samples

def load_dataset(ds: DatasetConfig, nsamples: int, tokenizer, seed: int, seqlen: int) -> Tuple[List, any]:
    """載入單一資料集"""
    if ds.name == "json" and ds.file_path:
        return get_loaders(
            "json", 
            nsamples=nsamples,
            seed=ds.file_path,  # 傳入檔案路徑
            seqlen=ds.max_length or seqlen,
            tokenizer=tokenizer
        )
    else:
        return get_loaders(
            ds.name,
            nsamples=nsamples,
            seed=seed,
            seqlen=ds.max_length or seqlen,
            tokenizer=tokenizer
        )

def mix_samples(all_samples: List, mixing_strategy: str, datasets: List[DatasetConfig]) -> List:
    """根據策略混合樣本"""
    if mixing_strategy == "weighted_random":
        # 根據權重隨機選擇
        weights = [sample["weight"] for sample in all_samples]
        return random.choices(all_samples, weights=weights, k=len(all_samples))
    
    elif mixing_strategy == "sequential":
        # 按順序排列
        return all_samples
    
    elif mixing_strategy == "interleaved":
        # 交錯排列，確保每個資料集都有代表性
        mixed = []
        max_samples = max(len([s for s in all_samples if s["source"] == ds.name]) 
                        for ds in datasets)
        
        for i in range(max_samples):
            for ds in datasets:
                ds_samples = [s for s in all_samples if s["source"] == ds.name]
                if i < len(ds_samples):
                    mixed.append(ds_samples[i])
        
        return mixed
    
    elif mixing_strategy == "balanced":
        # 平衡混合，確保每個資料集都有均等的代表性
        mixed = []
        dataset_groups = {}
        
        # 按資料集分組
        for sample in all_samples:
            source = sample["source"]
            if source not in dataset_groups:
                dataset_groups[source] = []
            dataset_groups[source].append(sample)
        
        # 交錯取樣
        max_samples = max(len(samples) for samples in dataset_groups.values())
        for i in range(max_samples):
            for samples in dataset_groups.values():
                if i < len(samples):
                    mixed.append(samples[i])
        
        return mixed
    
    else:
        return all_samples

def convert_to_wanda_format(mixed_samples: List) -> List:
    """轉換為WANDA標準格式"""
    # WANDA期望的格式是 (input_ids, target) 的元組列表
    wanda_samples = []
    for sample in mixed_samples:
        wanda_samples.append(sample["data"])
    return wanda_samples

def save_hybrid_dataset(mixed_samples: List, dataset_stats: Dict, config: HybridDatasetConfig):
    """儲存混合資料集到檔案"""
    if not config.save_path:
        return
        
    try:
        # 檢查並創建目錄路徑
        save_dir = os.path.dirname(config.save_path)
        ensure_directory_exists(save_dir, "儲存目錄")
        
        # 準備儲存資料
        save_data = {
            "config": {
                "total_nsamples": config.total_nsamples,
                "seqlen": config.seqlen,
                "mixing_strategy": config.mixing_strategy,
                "seed": config.seed
            },
            "dataset_stats": dataset_stats,
            "samples": []
        }
        
        # 儲存樣本（只儲存文字，不儲存張量）
        for sample in mixed_samples:
            input_ids, target = sample["data"]
            # 將張量轉換為列表以便JSON序列化
            save_data["samples"].append({
                "input_ids": input_ids.tolist(),
                "target": target.tolist(),
                "source": sample["source"],
                "weight": sample["weight"]
            })
        
        # 儲存到檔案
        with open(config.save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"混合資料集已儲存到: {config.save_path}")
        
    except Exception as e:
        print(f"儲存混合資料集時發生錯誤: {e}")
        print(f"嘗試創建目錄: {os.path.dirname(config.save_path) if config.save_path else 'N/A'}")
        raise

def create_hybrid_dataset(config: HybridDatasetConfig, model_path: str, tokenizer: Optional[object] = None) -> Tuple[List, Dict]:
    """創建混合資料集，返回標準的WANDA格式
    
    Args:
        config: 混合資料集配置
        model_path: 模型路徑，用於自動載入tokenizer
        tokenizer: 可選的tokenizer，如果不提供則從model_path自動載入
    
    Returns:
        Tuple[List, Dict]: (混合資料集樣本, 統計資訊)
    """
    # 設定隨機種子
    set_seed(config.seed)
    
    # 檢查並創建必要的目錄路徑
    if config.save_path:
        save_dir = os.path.dirname(config.save_path)
        ensure_directory_exists(save_dir, "儲存目錄")
    
    # 自動載入tokenizer（如果沒有提供）
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    all_samples = []
    dataset_stats = {}
    
    # 計算每個資料集的樣本數量
    dataset_samples = calculate_sample_distribution(config)
    
    # 從每個資料集載入資料
    for ds in config.datasets:
        print(f"載入資料集: {ds.name}, 樣本數: {dataset_samples[ds.name]}")
        
        try:
            # 載入資料集
            train_loader, _ = load_dataset(ds, dataset_samples[ds.name], tokenizer, config.seed, config.seqlen)
            
            # 記錄統計資訊
            dataset_stats[ds.name] = {
                "samples_loaded": len(train_loader),
                "weight": ds.weight,
                "max_length": ds.max_length or config.seqlen
            }
            
            # 將樣本加入總列表
            for sample in train_loader:
                all_samples.append({
                    "data": sample,
                    "source": ds.name,
                    "weight": ds.weight
                })
                
        except Exception as e:
            print(f"載入資料集 {ds.name} 時發生錯誤: {e}")
            continue
    
    # 根據混合策略重新排列樣本
    mixed_samples = mix_samples(all_samples, config.mixing_strategy, config.datasets)
    
    # 轉換為WANDA標準格式
    wanda_format_samples = convert_to_wanda_format(mixed_samples)
    
    # 儲存混合資料集（如果指定了儲存路徑）
    save_hybrid_dataset(mixed_samples, dataset_stats, config)
    
    return wanda_format_samples, dataset_stats

# 為了向後兼容，保留原有的函數簽名
def create_hybrid_dataset_legacy(config: HybridDatasetConfig, tokenizer):
    """向後兼容的函數名稱，使用外部提供的tokenizer"""
    print("警告: 使用舊版函數簽名，建議使用 create_hybrid_dataset(config, model_path)")
    # 這裡需要一個預設的模型路徑，或者拋出錯誤
    raise ValueError("請使用新的函數簽名: create_hybrid_dataset(config, model_path)")

