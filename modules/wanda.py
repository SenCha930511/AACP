# modules/wanda.py
import os   
import numpy as np
import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer
from tqdm import tqdm
from lib.model_utils import get_llm, find_layers, check_sparsity
from lib.calibration_core import prepare_calibration_input
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders
from config import PruningConfig
import gc

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdim=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def finalize_pruned_model(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # 你也可以指定 Conv2d 等
            try:
                prune.remove(module, 'weight')  # 把 pruning mask 真的移除，直接 apply 剪枝
            except ValueError:
                continue

def safe_forward_pass(layer, hidden_states, attention_mask=None, position_ids=None):
    """
    安全的前向傳播，處理不同版本的 transformers 兼容性問題
    """
    try:
        # 檢查輸入是否為None
        if hidden_states is None:
            raise ValueError("hidden_states cannot be None")
        
        # 確保所有張量都在相同設備上
        device = hidden_states.device
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if position_ids is not None and position_ids.device != device:
            position_ids = position_ids.to(device)
        
        # 檢查並修正attention_mask的形狀
        if attention_mask is not None:
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)
            
            # 如果attention_mask是4D且batch維度不匹配，修正它
            if attention_mask.dim() == 4:
                if attention_mask.size(0) != batch_size:
                    # 取第一個樣本並擴展到正確的batch size
                    attention_mask = attention_mask[0:1].expand(batch_size, -1, -1, -1)
            elif attention_mask.dim() == 2:
                if attention_mask.size(0) != batch_size:
                    # 取第一個樣本並擴展到正確的batch size
                    attention_mask = attention_mask[0:1].expand(batch_size, -1)
        
        # 檢查並修正position_ids的形狀
        if position_ids is not None:
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)
            
            if position_ids.dim() == 2:
                if position_ids.size(0) != batch_size:
                    # 取第一個樣本並擴展到正確的batch size
                    position_ids = position_ids[0:1].expand(batch_size, -1)
                if position_ids.size(1) != seq_len:
                    # 如果sequence長度不匹配，重新創建
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        # 嘗試完整的前向傳播
        if attention_mask is not None and position_ids is not None:
            result = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        elif attention_mask is not None:
            result = layer(hidden_states, attention_mask=attention_mask)
        else:
            result = layer(hidden_states)
        
        return result
        
    except Exception as e:
        print(f"Forward pass failed with full parameters: {e}")
        try:
            # 嘗試僅使用hidden_states
            return layer(hidden_states)
        except Exception as e2:
            print(f"Forward pass failed with minimal parameters: {e2}")
            try:
                # 嘗試使用use_cache=False
                return layer(hidden_states, use_cache=False)
            except Exception as e3:
                print(f"Forward pass failed with use_cache=False: {e3}")
                # 最後嘗試：返回輸入本身（跳過這一層）
                print("Warning: 跳過此層，返回輸入張量")
                return hidden_states

def prune_wanda(config: PruningConfig, model_path: str):
    # 合併環境變數設置，避免覆寫
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True,max_split_size_mb:32"

    print(f"Torch Version: {torch.__version__}")
    print(f"# of GPUs: {torch.cuda.device_count()}")

    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    prune_n, prune_m = 0, 0
    if config.sparsity_type != "unstructured":
        assert config.sparsity_ratio == 0.5, "N:M structured sparsity must be 0.5"
        prune_n, prune_m = map(int, config.sparsity_type.split(":"))

    print(f"Loading model {config.model}")
    model = get_llm(config.model, model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in config.model or "65b" in config.model:
        device = model.hf_device_map["lm_head"]

    print(f"Using device: {device}")

    if config.sparsity_ratio != 0:
        print("Pruning starts")
        dataloader, _ = get_loaders(config.dataset, nsamples=config.nsamples, seed=config.seed, seqlen=model.seqlen, tokenizer=tokenizer)
        print("Calibration data loaded")

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, batch_size=config.nsamples)

        # 確保 attention_mask 和 position_ids 的正確性
        if attention_mask is None:
            attention_mask = torch.ones((inps.size(0), model.seqlen), dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(inps.size(0), 1)

        # 調試信息
        print(f"[INFO] Input shapes: inps={inps.shape}, attention_mask={attention_mask.shape if attention_mask is not None else None}, position_ids={position_ids.shape if position_ids is not None else None}")

        layers = model.model.layers
        print(f"[INFO] Start pruning. Number of layers: {len(layers)}")

        for i in tqdm(range(len(layers))):
            print(f"[INFO] Processing layer {i}")
            layer = layers[i]
            subset = find_layers(layer)

            # 設備映射處理
            current_device = device
            if f"model.layers.{i}" in model.hf_device_map:
                current_device = model.hf_device_map[f"model.layers.{i}"]
                inps = inps.to(current_device)
                outs = outs.to(current_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(current_device)
                if position_ids is not None:
                    position_ids = position_ids.to(current_device)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # 前向傳播循環，含例外處理和詳細錯誤輸出
            for j in range(config.nsamples):
                with torch.no_grad():
                    try:
                        current_input = inps[j].unsqueeze(0)
                        current_attention_mask = None
                        current_position_ids = None

                        # 更安全的處理attention_mask
                        if attention_mask is not None:
                            if attention_mask.dim() == 2:
                                # 2D attention_mask: (batch_size, seq_len)
                                if j < attention_mask.size(0):
                                    current_attention_mask = attention_mask[j:j+1]
                                else:
                                    current_attention_mask = attention_mask[0:1]
                            elif attention_mask.dim() == 4:
                                # 4D attention_mask: (batch_size, num_heads, seq_len, seq_len)
                                if j < attention_mask.size(0):
                                    current_attention_mask = attention_mask[j:j+1]
                                else:
                                    current_attention_mask = attention_mask[0:1]
                            else:
                                # 其他情況，直接使用
                                current_attention_mask = attention_mask

                        # 更安全的處理position_ids
                        if position_ids is not None:
                            if position_ids.dim() == 2:
                                # 2D position_ids: (batch_size, seq_len)
                                if j < position_ids.size(0):
                                    current_position_ids = position_ids[j:j+1]
                                else:
                                    current_position_ids = position_ids[0:1]
                            else:
                                # 其他情況，直接使用
                                current_position_ids = position_ids

                        # 確保所有張量都不是None並且在正確設備上
                        if current_input is None:
                            print(f"[WARNING] current_input is None at layer {i}, sample {j}")
                            current_input = inps[j].unsqueeze(0)
                        
                        # 安全前向傳播
                        layer_output = safe_forward_pass(
                            layer, 
                            current_input, 
                            current_attention_mask, 
                            current_position_ids
                        )

                        # 確保layer_output不是None
                        if layer_output is None:
                            print(f"[WARNING] layer_output is None at layer {i}, sample {j}")
                            layer_output = current_input
                        
                        if isinstance(layer_output, tuple):
                            outs[j] = layer_output[0]
                        else:
                            outs[j] = layer_output

                    except Exception as e:
                        print(f"[ERROR] Forward pass error at layer {i}, sample {j}: {e}")
                        print(f"Input shape: {current_input.shape if current_input is not None else 'None'}")
                        print(f"Attention mask shape: {current_attention_mask.shape if current_attention_mask is not None else 'None'}")
                        print(f"Position IDs shape: {current_position_ids.shape if current_position_ids is not None else 'None'}")
                        
                        # 確保我們有一個有效的輸出
                        if current_input is not None:
                            outs[j] = current_input.squeeze(0)
                        else:
                            # 如果連current_input都是None，使用原始輸入
                            outs[j] = inps[j]

            # 移除 hooks
            for h in handles:
                h.remove()

            # 剪枝每個子層
            for name in subset:
                if name not in wrapped_layers:
                    continue
                try:
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    W_mask = (torch.zeros_like(W_metric) == 1)

                    if prune_n != 0:
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii:(ii + prune_m)].float()
                                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                    else:
                        try:
                            sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        except RuntimeError as e:
                            print(f"[ERROR] Sort failed for layer {i} name {name} with error: {e}")
                            continue

                        if config.use_variant:
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)
                            alpha = 0.4
                            alpha_hist = [0., 0.8]
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                            while (torch.abs(cur_sparsity - config.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > config.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha
                                alpha = alpha_new 
                                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                            print(f"[INFO] alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            indices = sort_res[1][:, :int(W_metric.shape[1] * config.sparsity_ratio)]
                            W_mask.scatter_(1, indices, True)

                    # 將 mask 為 True 的位置設為 0
                    subset[name].weight.data[W_mask] = 0

                    del W_metric, sort_res

                except Exception as e:
                    print(f"[ERROR] Pruning failed for layer {i} name {name}: {e}")
                    continue

            del wrapped_layers
            del handles

            # 交換輸入輸出，為下一層做準備
            inps, outs = outs, inps

            gc.collect()
            torch.cuda.empty_cache()

            print(f"[INFO] Completed processing layer {i}")
            print(f"[MEM] Memory allocated after layer {i}: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

    overall_sparsity, layer_sparsity = check_sparsity(model)
    print("*" * 30)
    print(f"Overall sparsity: {overall_sparsity:.4f}")

    if config.save:
        if not os.path.exists(config.save):
            os.makedirs(config.save)
        save_filepath = os.path.join(config.save, f"log_wanda.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"wanda\t{overall_sparsity:.4f}", file=f, flush=True)

    if config.save_model:
        finalize_pruned_model(model)
        model.save_pretrained(config.save_model)
        tokenizer.save_pretrained(config.save_model)
 
    return overall_sparsity