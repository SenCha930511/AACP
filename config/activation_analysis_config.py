from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ActivationAnalysisConfig:
    model: object  # HF model instance
    tokenizer: object  # HF tokenizer instance
    dataset_texts: List[str]  # list of text samples
    top_k: int = 60
    layer_index: int = 0
    max_length: int = 512
    device: str = 'cuda'
    save_path: Optional[str] = None
