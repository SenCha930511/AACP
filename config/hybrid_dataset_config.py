from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DatasetConfig:
    """單一資料集配置"""
    name: str  # 資料集名稱 (wikitext2, c4, ptb, openwebtext, dclm, json)
    weight: float = 1.0  # 在混合資料集中的權重
    nsamples: Optional[int] = None  # 該資料集的樣本數量
    max_length: Optional[int] = None  # 該資料集的最大長度限制
    file_path: Optional[str] = None  # 如果是json檔案，指定路徑

@dataclass
class HybridDatasetConfig:
    """混合資料集配置 - 專門針對WANDA剪枝優化"""
    datasets: List[DatasetConfig]  # 要混合的資料集列表
    total_nsamples: int = 128  # 總樣本數量
    seed: int = 0  # 隨機種子
    seqlen: int = 2048  # 序列長度
    mixing_strategy: str = "weighted_random"  # 混合策略
    save_path: Optional[str] = None  # 儲存混合資料集的路徑
    # WANDA特定配置
    balance_activation_coverage: bool = True  # 是否平衡啟用覆蓋率
    min_samples_per_dataset: int = 10  # 每個資料集的最小樣本數
