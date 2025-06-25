import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Dictionary to register dataset loader functions
DATASET_LOADERS = {}

# Decorator to register a dataset loader
def register_loader(name):
    def decorator(func):
        DATASET_LOADERS[name] = func
        return func
    return decorator

@register_loader("wikitext2")
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

@register_loader("c4")
def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

@register_loader("ptb")
def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

@register_loader("openwebtext")
def get_openwebtext(nsamples, seed, seqlen, tokenizer):
    # Use 10k subset (already small like c4 approach)
    dataset = load_dataset("stas/openwebtext-10k", split="train[:1000]")  # Take only first 1000 samples
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            enc = tokenizer(dataset[i]['text'], return_tensors='pt')
            if enc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # For eval: use a subset
    eval_text = " ".join(dataset[i]['text'] for i in range(min(100, len(dataset))))
    testenc = tokenizer(eval_text, return_tensors='pt')
    return trainloader, testenc

@register_loader("wikipedia")
def get_wikipedia(nsamples, seed, seqlen, tokenizer):
    # Only load first 10000 articles to speed up loading (like c4 approach)
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]")
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            text = dataset[i]['text']
            if len(text) > 50:  # Ensure text is not too short
                enc = tokenizer(text, return_tensors='pt')
                if enc.input_ids.shape[1] > seqlen:
                    break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # For eval: use a subset for evaluation
    eval_texts = [dataset[i]['text'] for i in range(min(100, len(dataset))) if len(dataset[i]['text']) > 50]
    eval_text = " ".join(eval_texts[:10])  # Use first 10 texts for evaluation
    testenc = tokenizer(eval_text, return_tensors='pt')
    return trainloader, testenc

@register_loader("slimapajama")
def get_slimapajama(nsamples, seed, seqlen, tokenizer):
    # Use streaming and take only first few samples (like c4 approach)
    dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    
    random.seed(seed)
    trainloader = []
    dataset_iter = iter(dataset)
    
    # Pre-load a small buffer of samples to work with
    sample_buffer = []
    for _ in range(min(1000, nsamples * 10)):  # Load limited samples
        try:
            sample = next(dataset_iter)
            if len(sample['text']) > 100:
                sample_buffer.append(sample['text'])
        except StopIteration:
            break
    
    if not sample_buffer:
        raise ValueError("No valid samples found in SlimPajama dataset")
    
    for _ in range(nsamples):
        while True:
            text = random.choice(sample_buffer)
            enc = tokenizer(text, return_tensors='pt')
            if enc.input_ids.shape[1] > seqlen:
                break
        
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # For eval: use samples from buffer
    eval_text = " ".join(sample_buffer[:5]) if sample_buffer else "Sample evaluation text."
    testenc = tokenizer(eval_text, return_tensors='pt')
    return trainloader, testenc

@register_loader("dclm")
def get_dclm(nsamples, seed, seqlen, tokenizer):
    # Use streaming and take only first few samples (like c4 approach)
    dataset = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)
    
    random.seed(seed)
    trainloader = []
    dataset_iter = iter(dataset)
    
    # Pre-load a small buffer of samples to work with
    sample_buffer = []
    for _ in range(min(1000, nsamples * 10)):  # Load limited samples
        try:
            sample = next(dataset_iter)
            if len(sample['text']) > 100:
                sample_buffer.append(sample['text'])
        except StopIteration:
            break
    
    if not sample_buffer:
        raise ValueError("No valid samples found in DCLM dataset")
    
    for _ in range(nsamples):
        while True:
            text = random.choice(sample_buffer)
            enc = tokenizer(text, return_tensors='pt')
            if enc.input_ids.shape[1] > seqlen:
                break
        
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # For eval: use samples from buffer
    eval_text = " ".join(sample_buffer[:5]) if sample_buffer else "Sample evaluation text."
    testenc = tokenizer(eval_text, return_tensors='pt')
    return trainloader, testenc

# Unified loader entry point
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    for key in DATASET_LOADERS:
        if key in name:
            return DATASET_LOADERS[key](nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Dataset '{name}' not supported. Available: {list(DATASET_LOADERS.keys())}")
