#!/usr/bin/env python3
import os
import argparse
import glob
import time
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size

def main():
    parser = argparse.ArgumentParser(description="Run inference for Needle-in-a-Haystack.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype_map[args.dtype],trust_remote_code=True).to(device)
    model.eval()

    ds = load_from_disk(args.dataset_path)
    indices = list(range(rank, len(ds), world_size))
    
    results = []
    for i in tqdm(indices, desc=f"Rank {rank} Inference"):
        prompt = ds[i]["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        gen_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append({**ds[i], "generated": gen_text})

    os.makedirs(args.output_dir, exist_ok=True)
    shard_path = os.path.join(args.output_dir, f"part-rank{rank:02d}.parquet")
    pd.DataFrame(results).to_parquet(shard_path)
    
    if world_size > 1: torch.distributed.barrier()

    if rank == 0:
        while len(glob.glob(os.path.join(args.output_dir, "part-rank*.parquet"))) < world_size:
            time.sleep(2)
        
        merged_df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(args.output_dir, "part-rank*.parquet")))])
        merged_df.to_parquet(os.path.join(args.output_dir, "all_results.parquet"))

if __name__ == "__main__":
    main()