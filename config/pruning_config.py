from dataclasses import dataclass
from typing import Optional

# 建立參數物件
@dataclass
class PruningConfig:
    seed: int = 0
    nsamples: int = 128
    sparsity_ratio: float = 0.3
    sparsity_type: str = "unstructured"
    model: str = "meta-llama/Llama-2-7b-hf"
    use_variant: bool = False
    dataset: str = "hybrid_coverage.jsonl"
    save: Optional[str] = None
    save_model: Optional[str] = "out/hybrid_coverage_30percent"
#topk_c4_activation.jsonl
