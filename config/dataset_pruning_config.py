from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetPruningConfig:
    """Dataset Pruning 配置類別
    
    基於論文 "Dataset Pruning: Reducing Training Data by Examining Generalization Influence"
    實現的配置參數
    """
    
    # 基本設定
    seed: int = 42
    target_samples: int = 64  # 目標樣本數量（剪枝後的樣本數）
    influence_threshold: float = 0.01  # 影響力閾值
    
    # 迭代設定
    max_iterations: int = 10  # 最大迭代次數
    convergence_threshold: float = 0.001  # 收斂閾值
    
    # 批次處理
    batch_size: int = 8  # 批次大小
    validation_batch_size: int = 16  # 驗證批次大小
    
    # 設備設定
    device: str = "cuda"  # 設備類型
    use_mixed_precision: bool = True  # 是否使用混合精度
    
    # 影響力計算設定
    influence_method: str = "cosine_similarity"  # 影響力計算方法
    similarity_metric: str = "cosine"  # 相似度度量方法
    importance_weight: float = 0.7  # 重要性權重
    diversity_weight: float = 0.3  # 多樣性權重
    
    # 驗證集設定
    validation_samples: int = 32  # 驗證樣本數量
    validation_split_ratio: float = 0.2  # 驗證集分割比例
    
    # 儲存設定
    save_path: Optional[str] = None  # 儲存路徑
    save_intermediate: bool = False  # 是否儲存中間結果
    save_format: str = "json"  # 儲存格式
    
    # 輸出設定
    verbose: bool = True  # 詳細輸出
    log_interval: int = 10  # 日誌間隔
    save_logs: bool = True  # 是否儲存日誌
    
    # 高級設定
    use_hessian: bool = False  # 是否使用Hessian矩陣計算影響力
    hessian_samples: int = 16  # Hessian計算樣本數
    use_gradient: bool = True  # 是否使用梯度信息
    gradient_norm: bool = True  # 是否正規化梯度
    
    # 多樣性設定
    diversity_threshold: float = 0.1  # 多樣性閾值
    max_similar_samples: int = 5  # 最大相似樣本數
    
    # 記憶體優化
    memory_efficient: bool = True  # 記憶體效率模式
    clear_cache_interval: int = 5  # 清理快取間隔
    
    def __post_init__(self):
        """驗證配置參數"""
        if self.target_samples <= 0:
            raise ValueError("target_samples 必須大於0")
        
        if self.influence_threshold < 0:
            raise ValueError("influence_threshold 必須大於等於0")
        
        if self.max_iterations <= 0:
            raise ValueError("max_iterations 必須大於0")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size 必須大於0")
        
        if self.validation_samples <= 0:
            raise ValueError("validation_samples 必須大於0")
        
        if self.importance_weight + self.diversity_weight != 1.0:
            print("警告: importance_weight + diversity_weight 應該等於1.0")
        
        if self.influence_method not in ["cosine_similarity", "euclidean_distance", "gradient_based", "hessian_based"]:
            raise ValueError("不支援的影響力計算方法")
        
        if self.similarity_metric not in ["cosine", "euclidean", "manhattan", "pearson"]:
            raise ValueError("不支援的相似度度量方法")

@dataclass
class AdvancedDatasetPruningConfig(DatasetPruningConfig):
    """進階Dataset Pruning配置，包含更多實驗性功能"""
    
    # 自適應設定
    adaptive_threshold: bool = True  # 自適應閾值
    adaptive_learning_rate: float = 0.1  # 自適應學習率
    
    # 集成方法
    ensemble_methods: bool = False  # 是否使用集成方法
    ensemble_size: int = 3  # 集成大小
    
    # 動態調整
    dynamic_sample_selection: bool = True  # 動態樣本選擇
    dynamic_threshold_adjustment: bool = True  # 動態閾值調整
    
    # 多目標優化
    multi_objective: bool = False  # 多目標優化
    objective_weights: tuple = (0.5, 0.3, 0.2)  # 目標權重
    
    # 時間預算
    time_budget: Optional[float] = None  # 時間預算（秒）
    early_stopping: bool = True  # 早停機制
    
    # 可解釋性
    explainable_selection: bool = False  # 可解釋的選擇
    feature_importance: bool = False  # 特徵重要性分析
    
    def __post_init__(self):
        """驗證進階配置參數"""
        super().__post_init__()
        
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size 必須大於0")
        
        if self.adaptive_learning_rate <= 0:
            raise ValueError("adaptive_learning_rate 必須大於0")
        
        if self.time_budget is not None and self.time_budget <= 0:
            raise ValueError("time_budget 必須大於0")
        
        if len(self.objective_weights) != 3:
            raise ValueError("objective_weights 必須包含3個值")
        
        if sum(self.objective_weights) != 1.0:
            print("警告: objective_weights 總和應該等於1.0")

# 預設配置
DEFAULT_DATASET_PRUNING_CONFIG = DatasetPruningConfig()
DEFAULT_ADVANCED_CONFIG = AdvancedDatasetPruningConfig()

# 快速配置函數
def create_fast_config(target_samples: int = 32) -> DatasetPruningConfig:
    """創建快速配置，適合快速實驗"""
    return DatasetPruningConfig(
        target_samples=target_samples,
        max_iterations=5,
        batch_size=16,
        validation_samples=16,
        verbose=False,
        memory_efficient=True
    )

def create_balanced_config(target_samples: int = 48) -> DatasetPruningConfig:
    """創建平衡配置，平衡速度和精度"""
    return DatasetPruningConfig(
        target_samples=target_samples,
        max_iterations=10,
        batch_size=8,
        validation_samples=32,
        use_gradient=True,
        diversity_threshold=0.15,
        verbose=True
    )

def create_accurate_config(target_samples: int = 64) -> DatasetPruningConfig:
    """創建精確配置，適合高精度實驗"""
    return DatasetPruningConfig(
        target_samples=target_samples,
        max_iterations=20,
        batch_size=4,
        validation_samples=64,
        use_hessian=True,
        hessian_samples=32,
        verbose=True,
        save_intermediate=True
    )
