# lib/calibration_core.py
import torch
from torch import nn
from torch.cuda.amp import autocast


def prepare_calibration_input(model, dataloader, device, batch_size=1):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    torch.cuda.empty_cache()

    inps = torch.zeros((batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, attention_mask=None, position_ids=None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = attention_mask
            cache['position_ids'] = position_ids
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        for batch in dataloader:
            autocast_context = autocast(dtype=torch.float16) if dtype == torch.float16 else torch.no_grad()
            with autocast_context:
                input_ids = batch[0].to(device)
                attention_mask_batch = batch[1].to(device) if len(batch) > 1 else (input_ids != model.config.pad_token_id).long()
                position_ids_batch = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(input_ids.size(0), 1)
                model(input_ids, attention_mask=attention_mask_batch, position_ids=position_ids_batch)
    except ValueError:
        pass

    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    
    # 修正張量擴展邏輯
    def safe_expand_to_batch(tensor, target_batch_size, seq_len):
        if tensor is None:
            return None
        
        # 確保tensor在正確的設備上
        if tensor.device != device:
            tensor = tensor.to(device)
        
        # 處理不同維度的張量
        if tensor.dim() == 2:  # (batch_size, seq_len)
            if tensor.size(0) == target_batch_size:
                return tensor
            elif tensor.size(0) == 1:
                # 擴展batch維度
                return tensor.expand(target_batch_size, -1)
            else:
                # 取第一個樣本並重複
                return tensor[0:1].expand(target_batch_size, -1)
        
        elif tensor.dim() == 4:  # (batch_size, num_heads, seq_len, seq_len)
            if tensor.size(0) == target_batch_size:
                return tensor
            elif tensor.size(0) == 1:
                # 擴展batch維度
                return tensor.expand(target_batch_size, -1, -1, -1)
            else:
                # 取第一個樣本並重複
                return tensor[0:1].expand(target_batch_size, -1, -1, -1)
        
        elif tensor.dim() == 3:  # (batch_size, seq_len, hidden_size) 或其他3D張量
            if tensor.size(0) == target_batch_size:
                return tensor
            elif tensor.size(0) == 1:
                return tensor.expand(target_batch_size, -1, -1)
            else:
                return tensor[0:1].expand(target_batch_size, -1, -1)
        
        else:
            # 其他情況，嘗試簡單的擴展
            if tensor.size(0) == target_batch_size:
                return tensor
            else:
                # 創建正確形狀的張量
                new_shape = [target_batch_size] + list(tensor.shape[1:])
                return tensor[0:1].expand(new_shape)
    
    # 創建正確形狀的attention_mask和position_ids
    attention_mask = safe_expand_to_batch(cache['attention_mask'], batch_size, model.seqlen)
    position_ids = safe_expand_to_batch(cache['position_ids'], batch_size, model.seqlen)
    
    # 如果attention_mask仍然是None，創建一個默認的
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, model.seqlen), dtype=torch.long, device=device)
    
    # 如果position_ids仍然是None，創建一個默認的
    if position_ids is None:
        position_ids = torch.arange(model.seqlen, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
    print(f"[INFO] Calibration準備完成:")
    print(f"  - inps shape: {inps.shape}")
    print(f"  - attention_mask shape: {attention_mask.shape}")
    print(f"  - position_ids shape: {position_ids.shape}")
    print(f"  - device: {device}")
    
    return inps, outs, attention_mask, position_ids