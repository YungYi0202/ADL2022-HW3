from argparse import ArgumentParser, Namespace

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, MT5ForConditionalGeneration, T5TokenizerFast, set_seed, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from utils import *
from dataset import NewsSummaryDataset

def get_splits_n_filenames(args):
    splits = [TEST]
    filenames = dict()
    filenames[TEST] = args.test_file_path
    return splits, filenames

def dev_epoch(args, model, dev_loader, tokenizer, device):
    model.eval()
    titles, preds, ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            generate_ids = model.generate(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device), 
                max_length=args.title_max_len,
                min_length=10,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=2.5,
                early_stopping=args.early_stopping
            )
            decoded_result = tokenizer.batch_decode( generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            pred = list(map(lambda x:x.strip(),decoded_result))
            preds.extend(pred)
            if TITLE in batch:
                titles.extend(batch[TITLE])
            if ID in batch:
                ids.extend(batch[ID])

    cnt = 0
    for i in range(len(preds)):
        if preds[i] == "":
            cnt += 1
            log = "Detect empty prediction."
            if len(titles) > 0:
                log += f" Title: {titles[i]}"
            elif len(ids) > 0:
                log += f" ID: {ids[i]}"
            print(log)
            preds[i] = "*"

    print(f"Prediction empty cnt: {cnt}")
            
    return titles, preds, ids

def main(args):
    set_seed(args.seed)
    if args.enable_rl:
        print("Enable reinforcement learning")
        from torch.distributions import Categorical
        from torch.autograd import Variable
    
    # print(args)

    # Data
    SPLITS, filenames = get_splits_n_filenames(args)
    raw_data = {split: read_data(filenames[split], KEYS[split]) for split in SPLITS}
    
    # Tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_path)
    
    # Dataset
    dataset = {
        split: NewsSummaryDataset(raw_data[split], tokenizer, args.title_max_len, args.maintext_max_len, args.add_prefix) 
        for split in SPLITS
        }

    
    print("Start Testing...")
    test_loader = DataLoader(dataset[TEST], batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = MT5ForConditionalGeneration.from_pretrained(args.load_model_ckpt).to(device=args.device)
    # TEST
    _, preds, ids = dev_epoch(args, model, test_loader, tokenizer, args.device)
    # Output the prediction
    with open(args.output_file_path, "w", encoding="utf8") as fp:
        for pred ,i,in zip(preds, ids):
            json.dump({"title":pred, "id":i}, fp, ensure_ascii = False)
            fp.write("\n")
            
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--maintext_max_len", type=int, default=2048)
    parser.add_argument("--title_max_len", type=int, default=64)
    
    parser.add_argument("--tokenizer_path", type=str, default="google/mt5-small")

    # decoder
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--early_stopping", action="store_true")

    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    parser.add_argument("--fp16_training", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Mode
    parser.add_argument(
        "--load_model_ckpt", 
        type=Path, 
        required=True
    )
    parser.add_argument("--add_prefix", action="store_true")
    
    # For TA to run
    parser.add_argument(
        "--test_file_path",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output_file_path",
        type=Path,
        required=True
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}")
    main(args)
