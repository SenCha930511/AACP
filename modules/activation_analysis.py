import torch
import random
import numpy as np
from tqdm import tqdm
import json
from config.activation_analysis_config import ActivationAnalysisConfig


def compare_activation_coverage(config: ActivationAnalysisConfig):
    """
    比較 activation-rich 校準資料與隨機選樣的啟用神經元比例，並儲存 top-k 的樣本至 JSON 檔案。

    Args:
        config: ActivationAnalysisConfig 參數物件

    Returns:
        Dict 統計資訊，包括各類樣本的 activation 覆蓋率
    """
    model = config.model
    tokenizer = config.tokenizer
    dataset_texts = config.dataset_texts

    model.eval()
    #model.to(config.device)

    activation_scores = []
    encoded_inputs = []

    def get_hook():
        def hook(module, input, output):
            # 取第 0 項作為真正的 activation tensor
            if isinstance(output, tuple):
                output = output[0]
            ratio = (output > 0).float().mean().item()
            activation_scores.append(ratio)
        return hook

    handle = model.model.layers[config.layer_index].register_forward_hook(get_hook())

    print("Forwarding all inputs to collect activation scores...")
    for text in tqdm(dataset_texts):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=config.max_length).to(config.device)
        encoded_inputs.append(inputs)
        with torch.no_grad():
            _ = model(**inputs)

    handle.remove()
    activation_scores = np.array(activation_scores)

    indices = np.argsort(activation_scores)[::-1]
    topk_indices = indices[:config.top_k]
    randk_indices = random.sample(range(len(dataset_texts)), config.top_k)

    topk_scores = activation_scores[topk_indices]
    randk_scores = activation_scores[randk_indices]

    if config.save_path is not None:
        print(f"Saving top-{config.top_k} activation-rich samples to {config.save_path}")
        topk_texts = [dataset_texts[i] for i in topk_indices]
        with open(config.save_path, 'w', encoding='utf-8') as f:
            for item in topk_texts:
                f.write(json.dumps({"text": item}, ensure_ascii=False) + "\n")

    return {
        "topk_mean": float(np.mean(topk_scores)),
        "topk_std": float(np.std(topk_scores)),
        "randk_mean": float(np.mean(randk_scores)),
        "randk_std": float(np.std(randk_scores)),
        "topk_scores": topk_scores.tolist(),
        "randk_scores": randk_scores.tolist(),
        "all_scores": activation_scores.tolist(),
    }
