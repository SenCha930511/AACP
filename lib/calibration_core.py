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
    
    # 以下是關鍵：強制讓 attention_mask 與 position_ids batch size 與 inps 一致
    
    # 注意：cache['attention_mask'] 可能是 2D 或 4D，要視具體維度決定怎麼 repeat / expand
    # 這裡假設 attention_mask shape 是 (batch_size, seq_len) 或 (batch_size, 1, seq_len, seq_len) 等
    
    def expand_to_batch(tensor, target_batch_size):
        if tensor is None:
            return None
        if tensor.size(0) == target_batch_size:
            return tensor
        else:
            # 利用 repeat 擴展 batch 維度
            repeat_times = [target_batch_size] + [1]*(tensor.dim()-1)
            return tensor.repeat(*repeat_times)
    
    attention_mask = expand_to_batch(cache['attention_mask'], batch_size)
    position_ids = expand_to_batch(cache['position_ids'], batch_size)
    
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print(f"inps shape: {inps.shape}, attention_mask shape: {attention_mask.shape}, position_ids shape: {position_ids.shape}")
    return inps, outs, attention_mask, position_ids
