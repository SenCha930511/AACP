import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math
import gc
import os

def evaluate_perplexity(
    model_path: str,
    dataset_name: str = "allenai/c4",  # 修復：使用正確的數據集名稱
    dataset_subset: str = "en",  # 修復：使用正確的子集名稱
    split: str = "validation",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 1024,
    num_samples: int = -1
):
    """
    評估大型語言模型在指定資料集上的困惑度。
    """
    print(f"正在載入模型：{model_path} 到 {device}...")
    
    # 載入模型和分詞器
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map="auto"  # 自動分配設備
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model.eval()

    # 確保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"正在載入資料集：{dataset_name} ({dataset_subset}) ({split} 分割)...")

    # 修復：使用正確的C4數據集載入方式
    try:
        if "c4" in dataset_name.lower():
            dataset = load_dataset(
                "allenai/c4", 
                name="en",  # 使用正確的name參數
                split=split, 
                streaming=True,
                trust_remote_code=False
            )
        else:
            dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"載入C4失敗，嘗試替代數據集: {e}")
        # 使用WikiText-2作為替代
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
        print("改用WikiText-2數據集")

    total_nll = 0.0
    total_tokens = 0
    processed_samples = 0

    print("開始計算困惑度...")

    # 處理streaming dataset
    dataset_iter = iter(dataset)
    pbar = tqdm(total=num_samples if num_samples != -1 else None, desc="Processing samples")

    try:
        while True:
            if num_samples != -1 and processed_samples >= num_samples:
                break

            try:
                item = next(dataset_iter)
            except StopIteration:
                break

            text = item["text"]
            if not text.strip():
                continue

            # 修復：正確處理編碼
            try:
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length*4)
                seq_len = encodings.input_ids.size(1)
                
                if seq_len < 2:  # 至少需要2個token
                    continue

                input_ids = encodings.input_ids.to(device)

                # 將長序列分割成塊處理
                chunk_nlls = []
                for j in range(0, seq_len, max_length):
                    begin_loc = j
                    end_loc = min(j + max_length, seq_len)
                    trg_len = end_loc - begin_loc

                    if trg_len <= 1:  # 需要至少2個token來計算loss
                        continue

                    input_ids_chunk = input_ids[:, begin_loc:end_loc]
                    
                    # 修復：正確設置labels
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids_chunk, labels=input_ids_chunk)
                        # 修復：正確計算negative log likelihood
                        chunk_nll = outputs.loss.item() * (trg_len - 1)  # 減1因為最後一個token沒有預測目標
                        chunk_nlls.append(chunk_nll)
                        total_tokens += (trg_len - 1)

                    # 清理記憶體
                    del input_ids_chunk, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 累加這個樣本的總NLL
                total_nll += sum(chunk_nlls)
                processed_samples += 1
                
                # 更新進度條
                pbar.update(1)
                
                # 每處理10個樣本顯示當前困惑度
                if processed_samples % 10 == 0 and total_tokens > 0:
                    current_ppl = math.exp(total_nll / total_tokens)
                    pbar.set_postfix({"PPL": f"{current_ppl:.2f}", "Tokens": total_tokens})

            except Exception as e:
                print(f"處理樣本時出錯: {e}")
                continue

    except KeyboardInterrupt:
        print("\n評估被用戶中斷")
    finally:
        pbar.close()

    if total_tokens == 0:
        print("沒有足夠的有效token來計算困惑度。")
        return None

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    print(f"\n模型：{model_path}")
    print(f"處理樣本數：{processed_samples}")
    print(f"總token數：{total_tokens}")
    print(f"平均負對數似然 (NLL)：{avg_nll:.4f}")
    print(f"困惑度 (Perplexity)：{perplexity:.4f}")

    return perplexity

def quick_perplexity_test(model_path: str, num_samples: int = 50):
    """
    快速困惑度測試，使用WikiText-2
    """
    print(f"--- 快速困惑度測試 (WikiText-2, {num_samples} 樣本) ---")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 使用WikiText-2進行快速測試
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, total=min(num_samples, len(dataset)))):
                if i >= num_samples:
                    break
                    
                text = example['text']
                if len(text.strip()) < 10:  # 跳過太短的文本
                    continue
                
                inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                if inputs.input_ids.size(1) < 2:
                    continue
                
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs['input_ids'])
                
                loss = outputs.loss.item()
                seq_len = inputs['input_ids'].size(1) - 1  # 減1因為最後一個token沒有預測目標
                
                total_loss += loss * seq_len
                total_tokens += seq_len
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            print(f"困惑度 (WikiText-2): {perplexity:.2f}")
            return perplexity
        else:
            print("沒有有效的tokens")
            return None
            
    except Exception as e:
        print(f"快速測試失敗: {e}")
        return None

if __name__ == "__main__":
    # 設置模型路徑
    base_output_dir = "out/hybrid_coverage_30percent"

    # 自動檢測out目錄下的模型
    if os.path.exists(base_output_dir):
        models_in_out = [d for d in os.listdir(base_output_dir) 
                        if os.path.isdir(os.path.join(base_output_dir, d))]
        
        if models_in_out:
            print(f"在out/目錄下發現模型: {models_in_out}")
            # 使用第一個找到的模型，或者你可以手動指定
            pruned_model_name = models_in_out[0]
        else:
            print("out/目錄下沒有找到模型資料夾")
            # 直接使用out作為模型路徑（如你的情況）
            pruned_model_name = "hybrid_coverage_30percent"
    else:
        print("out/目錄不存在")
        exit(1)
    
    model_full_path = os.path.join(base_output_dir, pruned_model_name) if pruned_model_name else base_output_dir
    model_full_path = "out/hybrid_coverage_30percent"    
    print(f"--- 評估模型 '{model_full_path}' ---")
    
    # 先進行快速測試
    quick_ppl = quick_perplexity_test(model_full_path, num_samples=20)
    
    # 如果快速測試成功，再進行C4測試
    if quick_ppl is not None:
        print(f"\n--- 評估模型在C4上的困惑度 ---")
        try:
            perplexity_pruned = evaluate_perplexity(
                model_path=model_full_path,
                dataset_name="allenai/c4",
                dataset_subset="en",
                split="validation",
                max_length=512,  # 降低以節省記憶體
                num_samples=100  # 增加到100個樣本獲得更準確的結果
            )
        except Exception as e:
            print(f"C4評估失敗: {e}")
            print("這可能是由於記憶體不足或網路問題")
