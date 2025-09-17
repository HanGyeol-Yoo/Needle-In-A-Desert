#!/usr/bin/env python3
import os
import json
import random
import argparse
import itertools
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

G_TOKENIZER, G_CTX, G_NEEDLES, G_LINES, G_USE_CHAT = None, None, {}, {}, False

def load_jsonl(path: str) -> List[dict]:
    recs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            recs.append(json.loads(line.strip()))
    return recs

def load_needles_grouped_by_language(needles_path: str) -> Dict[str, List[dict]]:
    grouped = {}
    for rec in load_jsonl(needles_path):
        lang = rec.get("language")
        if lang:
            grouped.setdefault(lang, []).append(rec)
    return grouped

class ContextBuilder:
    def __init__(self, tokenizer): self.tok = tokenizer
    def encode(self, text: str) -> List[int]: return self.tok.encode(text, add_special_tokens=False)
    def decode(self, ids: List[int]) -> str: return self.tok.decode(ids, skip_special_tokens=True)
    def insert_needle_at_depth(self, ctx_tokens: List[int], needle: str, depth: int) -> str:
        needle_ids = self.encode(needle)
        pos = int(len(ctx_tokens) * (depth / 100.0))
        return self.decode(ctx_tokens[:pos] + needle_ids + ctx_tokens[pos:])
    def build_context_from_corpus(self, lines: List[dict], target_len: int, rng: random.Random) -> List[int]:
        idxs, acc = list(range(len(lines))), []
        rng.shuffle(idxs)
        for i in idxs:
            text = lines[i].get("text", "")
            if text and len(acc) < target_len:
                acc.extend(self.encode(text))
        return acc[:target_len]

def make_prompt(lang: str, context: str, question: str, use_chat: bool = False, tokenizer=None) -> str:
    base_prompts = {
        "English": "This is a test of long-text understanding. Read the document and answer the question.\n\n<Document>\n{context}\n</Document>\n\nBased on the document, {question}",
        "Korean": "이것은 장문 이해 능력 평가 과제입니다. 아래 문서를 읽고 질문에 답하세요.\n\n<문서>\n{context}\n</문서>\n\n문서를 바탕으로, {question}"
    }
    
    prompt_text = base_prompts[lang].format(context=context, question=question)
    
    if use_chat and tokenizer and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return prompt_text

def _init_worker(tokenizer_model, en_lines, kr_lines, needles_by_lang, use_chat):
    global G_TOKENIZER, G_CTX, G_LINES, G_NEEDLES, G_USE_CHAT
    G_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
    G_CTX = ContextBuilder(G_TOKENIZER)
    G_LINES = {"English": en_lines, "Korean": kr_lines}
    G_NEEDLES = needles_by_lang
    G_USE_CHAT = use_chat

def _process_one_from_tuple(args):
    lang, length, depth, rep, seed = args
    rng = random.Random((hash((lang, length, depth, rep)) ^ seed))
    needle_rec = rng.choice(G_NEEDLES[lang])
    needle_ids = G_CTX.encode(needle_rec["needle"])
    target_ctx_len = max(0, length - len(needle_ids))
    ctx_ids = G_CTX.build_context_from_corpus(G_LINES[lang], target_ctx_len, rng)
    ctx_with_needle = G_CTX.insert_needle_at_depth(ctx_ids, needle_rec["needle"], depth)
    if G_USE_CHAT and hasattr(G_TOKENIZER, "chat_template") and G_TOKENIZER.chat_template:
        question = needle_rec["retrieval_question_QA"]
    else:
        question = needle_rec["retrieval_question"]
    prompt = make_prompt(lang, ctx_with_needle, question, G_USE_CHAT, G_TOKENIZER)
    lang_code = "en" if lang == "English" else "kr"
    return {"title": f"Length{length}Depth{depth}_{lang_code}_{rep}", "prompt": prompt, "answer": needle_rec["arg"]}

def main():
    parser = argparse.ArgumentParser(description="Build Needle-in-a-Haystack dataset.")
    parser.add_argument("--tokenizer_model", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--needles_path", type=str, required=True)
    parser.add_argument("--en_jsonl", type=str, required=True)
    parser.add_argument("--kr_jsonl", type=str, required=True)
    parser.add_argument("--len_start", type=int, default=4000)
    parser.add_argument("--len_end", type=int, default=128000)
    parser.add_argument("--len_step", type=int, default=8000)
    parser.add_argument("--repeats_per_combo", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--chat", action="store_true", help="Use chat template if available in tokenizer")
    args = parser.parse_args()

    nproc = args.nproc if args.nproc else cpu_count()
    needles = load_needles_grouped_by_language(args.needles_path)
    en_lines, kr_lines = load_jsonl(args.en_jsonl), load_jsonl(args.kr_jsonl)
    
    lengths = list(range(args.len_start, args.len_end + 1, args.len_step))
    depths = list(range(0, 101, 10))
    langs = [lang for lang in ["English", "Korean"] if lang in needles]
    
    tasks = list(itertools.product(langs, lengths, depths, range(args.repeats_per_combo), [args.base_seed]))
    
    with Pool(nproc, _init_worker, (args.tokenizer_model, en_lines, kr_lines, needles, args.chat)) as pool:
        results = list(tqdm(pool.imap(_process_one_from_tuple, tasks), total=len(tasks), desc="Generating samples"))
        
    ds = Dataset.from_list(results)
    os.makedirs(args.output_path, exist_ok=True)
    ds.save_to_disk(args.output_path)

if __name__ == "__main__":
    main()