# AACP - Activation-Aware Calibration Data Selection for Pruning  
  
## 專案簡介  
  
AACP 是一個專注於大型語言模型（LLM）剪枝的研究專案，實現了基於啟動感知校準資料選擇的剪枝演算法。本專案提供了模型剪枝、LoRA 微調、困惑度評估等完整功能。  
  
## 主要特性  
  
- **WANDA 剪枝**：實現權重與啟動值乘積的結構化/非結構化剪枝  
- **啟動分析**：分析並比較不同校準資料的神經元啟動覆蓋率  
- **LoRA 微調**：支援低秩適應（LoRA）的高效微調  
- **模型評估**：提供困惑度（Perplexity）評估工具  
- **稀疏度評估**：評估剪枝後模型的稀疏度  
  
## 系統需求  
  
### 硬體需求  
- GPU：建議使用 1 張以上 GPU  
- 記憶體：建議 32GB 以上  
  
### 軟體需求  
詳見 `requirements.txt`
  
## 安裝步驟  
  
1. 使用此專案：  
```bash  
git clone https://github.com/SenCha930511/AACP.git  
cd AACP
```

2. 安裝依賴套件：
```bash
pip install -r requirements.txt
```
## 專案結構

AACP/  
├── activation_aware_pruning.py  # 主要 API 類別  
├── eval_ppl.py                  # 困惑度評估工具  
├── job.sh                       # SLURM 作業腳本  
├── requirements.txt             # 相依套件清單  
├── config/                      # 配置檔案目錄  
│   ├── pruning_config.py       # 剪枝設定  
│   ├── lora_config.py          # LoRA 設定   
│   ├── wandapp_config.py       # WANDAPP 設定   
│   ├── evaluate_config.py      # 評估設定  
│   └── activation_analysis_config.py  # 啟動分析設定  
├── modules/                     # 功能模組  
│   ├── wanda.py                # WANDA 剪枝實現  
│   ├── wandapp.py              # WANDAPP 剪枝實現  
│   ├── lora.py                 # LoRA 微調  
│   ├── evaluate.py             # 模型評估  
│   ├── activation_analysis.py  # 啟動分析  
│   ├── global_coverage.py      # 全域覆蓋率分析  
│   └── sparsity_eval.py        # 稀疏度評估  
└── lib/                         # 核心函式庫  
    ├── data.py                 # 資料載入器  
    ├── model_utils.py          # 模型工具  
    ├── layerwrapper.py         # 層包裝器  
    ├── calibration_core.py     # 校準核心

## 使用方法
1. 基本使用範例
```python
from activation_aware_pruning import ActivationAwarePruning  
from config import PruningConfig  
  
# 初始化剪枝物件  [1](#header-1)
pruner = ActivationAwarePruning(model_path="meta-llama/Llama-2-7b-hf")  
  
# 設定剪枝配置  [2](#header-2)
config = PruningConfig(  
    seed=0,  
    nsamples=128,  
    sparsity_ratio=0.3,  
    sparsity_type="unstructured",  
    dataset="hybrid_coverage.jsonl",  
    save_model="out/pruned_model"  
)  
  
# 執行 WANDA 剪枝  
pruner.prune_wanda(config)
```
2. 困惑度評估
使用獨立的評估腳本：
```python
python eval_ppl.py
```
該腳本支援在 C4 和 WikiText-2 資料集上評估模型困惑度。

3. SLURM 集群使用
如果您在 HPC 集群環境中運行，可以使用提供的 SLURM 腳本
```bash
sbatch job.sh
```
## 支援的資料集
專案內建支援以下資料集：

C4：Common Crawl 的乾淨版本
WikiText-2：維基百科文章資料集
自訂 JSONL 格式：可使用自己的校準資料

## 剪枝演算法
本專案實現的核心剪枝方法基於權重與啟動值的乘積進行重要性評估

## 注意事項
記憶體管理：大型模型剪枝需要大量 GPU 記憶體，建議使用多 GPU 或調整 batch size
校準資料：校準資料的選擇會顯著影響剪枝效果，建議使用啟動分析功能選擇合適的資料
模型儲存：剪枝後的模型會自動儲存到指定路徑

## 相關研究
本專案實現了基於啟動感知的校準資料選擇方法，用於改善神經網路剪枝的效果。

## 貢獻指南
歡迎提交 Issue 和 Pull Request 來改進本專案。
