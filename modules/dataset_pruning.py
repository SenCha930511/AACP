import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from tqdm import tqdm
import json
import os
from dataclasses import dataclass
from lib.model_utils import get_llm
from lib.data import get_loaders
from transformers import AutoTokenizer
import gc

@dataclass
class DatasetPruningConfig:
    """Dataset Pruning 配置"""
    seed: int = 42
    target_samples: int = 64  # 目標樣本數量
    influence_threshold: float = 0.01  # 影響力閾值
    max_iterations: int = 10  # 最大迭代次數
    batch_size: int = 8  # 批次大小
    device: str = "cuda"  # 設備
    save_path: Optional[str] = None  # 儲存路徑
    verbose: bool = True  # 詳細輸出

class InfluenceCalculator:
    """計算樣本影響力的核心類別"""
    
    def __init__(self, model: nn.Module, config: DatasetPruningConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()
        
    def compute_sample_influence(self, 
                               sample: Tuple[torch.Tensor, torch.Tensor],
                               validation_samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """
        計算單一樣本對驗證集的影響力
        
        Args:
            sample: 單一樣本 (input_ids, target)
            validation_samples: 驗證樣本列表
            
        Returns:
            float: 影響力分數
        """
        input_ids, target = sample
        input_ids = input_ids.unsqueeze(0).to(self.device)
        target = target.unsqueeze(0).to(self.device)
        
        # 計算原始驗證損失
        original_losses = []
        with torch.no_grad():
            for val_input, val_target in validation_samples:
                val_input = val_input.unsqueeze(0).to(self.device)
                val_target = val_target.unsqueeze(0).to(self.device)
                
                outputs = self.model(val_input)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), val_target.view(-1))
                original_losses.append(loss.item())
        
        # 計算包含樣本後的驗證損失
        updated_losses = []
        with torch.no_grad():
            # 這裡我們使用一個簡化的方法來估計影響
            # 實際實現中可能需要更複雜的影響力計算
            
            # 計算樣本的重要性分數
            sample_outputs = self.model(input_ids)
            if isinstance(sample_outputs, tuple):
                sample_logits = sample_outputs[0]
            else:
                sample_logits = sample_outputs
            
            # 使用logits的方差作為重要性指標
            importance_score = torch.var(sample_logits).item()
            
            # 計算與驗證樣本的相似度
            similarity_scores = []
            for val_input, _ in validation_samples:
                val_input = val_input.unsqueeze(0).to(self.device)
                val_outputs = self.model(val_input)
                if isinstance(val_outputs, tuple):
                    val_logits = val_outputs[0]
                else:
                    val_logits = val_outputs
                
                # 計算cosine相似度
                similarity = torch.cosine_similarity(
                    sample_logits.view(-1), 
                    val_logits.view(-1), 
                    dim=0
                ).item()
                similarity_scores.append(similarity)
            
            # 綜合影響力分數
            avg_similarity = np.mean(similarity_scores)
            influence_score = importance_score * avg_similarity
            
        return influence_score
    
    def compute_generalization_influence(self, 
                                       samples: List[Tuple[torch.Tensor, torch.Tensor]],
                                       validation_samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[float]:
        """
        計算所有樣本的泛化影響力
        
        Args:
            samples: 訓練樣本列表
            validation_samples: 驗證樣本列表
            
        Returns:
            List[float]: 每個樣本的影響力分數
        """
        influence_scores = []
        
        if self.config.verbose:
            print("計算樣本影響力分數...")
        
        for i, sample in enumerate(tqdm(samples, desc="計算影響力")):
            try:
                influence = self.compute_sample_influence(sample, validation_samples)
                influence_scores.append(influence)
                
                if self.config.verbose and i % 10 == 0:
                    print(f"樣本 {i}: 影響力 = {influence:.6f}")
                    
            except Exception as e:
                print(f"計算樣本 {i} 影響力時發生錯誤: {e}")
                influence_scores.append(0.0)
        
        return influence_scores

class DatasetPruner:
    """Dataset Pruning 主要類別"""
    
    def __init__(self, config: DatasetPruningConfig):
        self.config = config
        self.influence_calculator = None
        
    def setup_model(self, model_path: str, model_name: str):
        """設置模型和影響力計算器"""
        print(f"載入模型: {model_name}")
        model = get_llm(model_name, model_path)
        
        # 載入tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.model = model
        self.tokenizer = tokenizer
        self.influence_calculator = InfluenceCalculator(model, self.config)
        
        print("模型設置完成")
        
    def create_validation_set(self, 
                             dataset_name: str, 
                             nsamples: int = 32) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """創建驗證集用於影響力計算"""
        print(f"創建驗證集: {dataset_name}, 樣本數: {nsamples}")
        
        try:
            dataloader, _ = get_loaders(
                dataset_name, 
                nsamples=nsamples, 
                seed=self.config.seed + 1,  # 使用不同的種子
                seqlen=self.model.seqlen, 
                tokenizer=self.tokenizer
            )
            
            validation_samples = []
            for sample in dataloader:
                validation_samples.append(sample)
                if len(validation_samples) >= nsamples:
                    break
                    
            print(f"驗證集創建完成，樣本數: {len(validation_samples)}")
            return validation_samples
            
        except Exception as e:
            print(f"創建驗證集失敗: {e}")
            # 創建一個簡單的隨機驗證集
            return self._create_random_validation_set(nsamples)
    
    def _create_random_validation_set(self, nsamples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """創建隨機驗證集作為備用"""
        print("創建隨機驗證集...")
        
        vocab_size = self.tokenizer.vocab_size
        seq_len = self.model.seqlen
        
        validation_samples = []
        for _ in range(nsamples):
            # 隨機生成input_ids
            input_ids = torch.randint(0, vocab_size, (seq_len,))
            target = input_ids.clone()
            
            validation_samples.append((input_ids, target))
            
        return validation_samples
    
    def prune_dataset(self, 
                     dataset_name: str, 
                     nsamples: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict]:
        """
        執行dataset pruning
        
        Args:
            dataset_name: 資料集名稱
            nsamples: 原始樣本數量
            
        Returns:
            Tuple: (剪枝後的樣本, 統計資訊)
        """
        print(f"開始Dataset Pruning: {dataset_name}")
        print(f"目標樣本數: {self.config.target_samples}")
        
        # 載入完整資料集
        try:
            dataloader, _ = get_loaders(
                dataset_name, 
                nsamples=nsamples, 
                seed=self.config.seed, 
                seqlen=self.model.seqlen, 
                tokenizer=self.tokenizer
            )
            
            all_samples = list(dataloader)
            print(f"載入完整資料集，樣本數: {len(all_samples)}")
            
        except Exception as e:
            print(f"載入資料集失敗: {e}")
            return [], {"error": str(e)}
        
        # 創建驗證集
        validation_samples = self.create_validation_set(dataset_name, nsamples=32)
        
        # 計算影響力分數
        influence_scores = self.influence_calculator.compute_generalization_influence(
            all_samples, validation_samples
        )
        
        # 根據影響力分數排序樣本
        sample_scores = list(zip(all_samples, influence_scores))
        sample_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 選擇最有價值的樣本
        selected_samples = [sample for sample, _ in sample_scores[:self.config.target_samples]]
        selected_scores = [score for _, score in sample_scores[:self.config.target_samples]]
        
        # 計算統計資訊
        stats = {
            "original_samples": len(all_samples),
            "pruned_samples": len(selected_samples),
            "reduction_ratio": 1 - len(selected_samples) / len(all_samples),
            "avg_influence_score": np.mean(selected_scores),
            "min_influence_score": np.min(selected_scores),
            "max_influence_score": np.max(selected_scores),
            "influence_scores": selected_scores
        }
        
        print(f"Dataset Pruning 完成!")
        print(f"原始樣本數: {stats['original_samples']}")
        print(f"剪枝後樣本數: {stats['pruned_samples']}")
        print(f"減少比例: {stats['reduction_ratio']:.2%}")
        print(f"平均影響力分數: {stats['avg_influence_score']:.6f}")
        
        # 儲存結果
        if self.config.save_path:
            self._save_pruned_dataset(selected_samples, stats)
        
        return selected_samples, stats
    
    def _save_pruned_dataset(self, samples: List[Tuple[torch.Tensor, torch.Tensor]], stats: Dict):
        """儲存剪枝後的資料集"""
        try:
            save_dir = os.path.dirname(self.config.save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 準備儲存資料
            save_data = {
                "config": {
                    "target_samples": self.config.target_samples,
                    "influence_threshold": self.config.influence_threshold,
                    "seed": self.config.seed
                },
                "stats": stats,
                "samples": []
            }
            
            # 儲存樣本（轉換為列表格式）
            for i, (input_ids, target) in enumerate(samples):
                save_data["samples"].append({
                    "index": i,
                    "input_ids": input_ids.tolist(),
                    "target": target.tolist(),
                    "influence_score": stats["influence_scores"][i]
                })
            
            # 儲存到檔案
            with open(self.config.save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"剪枝後的資料集已儲存到: {self.config.save_path}")
            
        except Exception as e:
            print(f"儲存剪枝資料集時發生錯誤: {e}")

def create_pruned_calibration_data(config: DatasetPruningConfig,
                                 model_path: str,
                                 model_name: str,
                                 dataset_name: str,
                                 nsamples: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict]:
    """
    創建用於WANDA剪枝的校準資料
    
    Args:
        config: Dataset Pruning 配置
        model_path: 模型路徑
        model_name: 模型名稱
        dataset_name: 資料集名稱
        nsamples: 原始樣本數量
        
    Returns:
        Tuple: (剪枝後的校準樣本, 統計資訊)
    """
    # 創建pruner實例
    pruner = DatasetPruner(config)
    
    # 設置模型
    pruner.setup_model(model_path, model_name)
    
    # 執行pruning
    pruned_samples, stats = pruner.prune_dataset(dataset_name, nsamples)
    
    return pruned_samples, stats

# 為了向後兼容，保留原有的函數簽名
def create_pruned_calibration_data_legacy(config: DatasetPruningConfig,
                                        model_path: str,
                                        model_name: str,
                                        dataset_name: str,
                                        nsamples: int):
    """向後兼容的函數名稱"""
    print("警告: 使用舊版函數簽名，建議使用 create_pruned_calibration_data")
    return create_pruned_calibration_data(config, model_path, model_name, dataset_name, nsamples)
