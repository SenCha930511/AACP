import torch
import numpy as np
import random
import json
from tqdm import tqdm
from config.activation_analysis_config import ActivationAnalysisConfig

def compare_global_activation_coverage(config: ActivationAnalysisConfig):
    """
    使用 greedy strategy 根據全層 activation coverage 計算 activation-rich 的樣本，
    並挑出互補性高、總 coverage 最大的 top-K calibration texts。

    Returns:
        Dict 統計資訊，包括 activation-rich 與隨機樣本的 coverage 指標。
    """
    model = config.model
    tokenizer = config.tokenizer
    dataset_texts = config.dataset_texts

    model.eval()
    num_layers = len(model.model.layers)
    hidden_size = model.model.layers[0].mlp.gate_proj.out_features
    total_neurons = num_layers * hidden_size

    # 每筆樣本 → 每層啟用 neuron 的 index set
    all_coverages = []  # List[List[Set[int]]] 長度 = num_samples，內層長度 = num_layers

    def get_hooks(coverage_sets):
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                active = (output > 0).float()
                active = active.max(dim=1)[0]  # [batch, hidden_size]
                act_indices = (active[0] > 0).nonzero(as_tuple=False).squeeze(-1).cpu().numpy().tolist()
                coverage_sets[layer_idx] = set(act_indices)
            return hook
        return [model.model.layers[i].register_forward_hook(make_hook(i)) for i in range(num_layers)]

    print("Forwarding all inputs to collect per-sample coverage info...")
    for text in tqdm(dataset_texts):
        coverage_sets = [set() for _ in range(num_layers)]
        hooks = get_hooks(coverage_sets)

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=config.max_length).to(config.device)
        with torch.no_grad():
            _ = model(**inputs)

        for h in hooks:
            h.remove()

        all_coverages.append(coverage_sets)

    # Greedy union selection — maximize novel neuron coverage
    selected_indices = []
    covered = [set() for _ in range(num_layers)]

    for _ in range(config.top_k):
        best_idx = -1
        best_gain = -1

        for i, coverage in enumerate(all_coverages):
            if i in selected_indices:
                continue
            gain = 0
            for l in range(num_layers):
                gain += len(coverage[l] - covered[l])
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        for l in range(num_layers):
            covered[l].update(all_coverages[best_idx][l])

    # 隨機 baseline
    randk_indices = random.sample(range(len(dataset_texts)), config.top_k)

    # 計算 coverage ratio
    def compute_total_coverage(index_list):
        temp_covered = [set() for _ in range(num_layers)]
        for idx in index_list:
            for l in range(num_layers):
                temp_covered[l].update(all_coverages[idx][l])
        total_active = sum(len(s) for s in temp_covered)
        return total_active / total_neurons

    topk_ratio = compute_total_coverage(selected_indices)
    randk_ratio = compute_total_coverage(randk_indices)

    # 儲存選出來的 sample
    if config.save_path:
        print(f"Saving top-{config.top_k} complementary activation-rich samples to {config.save_path}")
        with open(config.save_path, 'w', encoding='utf-8') as f:
            for idx in selected_indices:
                f.write(json.dumps({"text": dataset_texts[idx]}, ensure_ascii=False) + "\n")

    return {
        "topk_coverage_ratio": topk_ratio,
        "randk_coverage_ratio": randk_ratio,
        "selected_indices": selected_indices,
        "random_indices": randk_indices,
    }

