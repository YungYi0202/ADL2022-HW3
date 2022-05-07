from argparse import ArgumentParser, Namespace

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, MT5ForConditionalGeneration, T5Tokenizer, set_seed, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from utils import *
from dataset import NewsSummaryDataset

from tw_rouge import get_rouge
import csv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_splits_n_filenames(args):
    splits = []
    filenames = dict()
    if not args.no_train:
        splits.extend([TRAIN, DEV])
        filenames[TRAIN] = args.train_file_path
        filenames[DEV] = args.dev_file_path
    if not args.no_test:
        splits.append(TEST)
        filenames[TEST] = args.test_file_path
    return splits, filenames

# Not sure if this is workable.
def policy_grad(args, loss, logits, tokenizer, titles):
    policys = None
    all_rewards = []
    for logit, title in zip (logits, titles):
        actions = []
        for idx, logit in enumerate(logit):
            d = Categorical(logits=logit)
            act = d.sample()
            actions.append(act)
            if policys == None:
                # First
                policys = d.log_prob(act).unsqueeze(0)
            else:
                policys = torch.cat([policys, d.log_prob(act).unsqueeze(0)])
            if actions[-1] == 1 or idx >= args.title_max_len:
                break
        actions = tokenizer.decode(torch.tensor(actions).to(args.device), skip_special_tokens = True, clean_up_tokenization_spaces = True)
        R = 0 if not actions else get_rouge(actions, title)["rouge-l"]["f"]
        rewards = []
        for _ in range(policys.shape[0]):
            # implement baseline = 0.2
            rewards.insert(0, R - 0.2)
            R *= args.gamma
        all_rewards.extend(rewards)
        print(f"Actions: {actions}, Original Loss: {loss.item()}")
        
    all_rewards = torch.tensor(all_rewards).to(args.device)
    # loss = total reward * (-1)
    loss = torch.sum(torch.mul(policys, Variable(all_rewards)).mul(-1),-1)
    return loss

def train_epoch(epoch, args, model, train_loader, optimizer, scheduler, device):
    model.train()
    
    train_loss = 0.0
    train_losses = []
        
    for step, batch in enumerate(tqdm(train_loader)):	                 
        output = model(batch["input_ids"].to(device), 
                    attention_mask=batch["attention_mask"].to(device), 
                    labels=batch["labels"].to(device), 
                    decoder_attention_mask=batch["labels_attention_mask"].to(device))

        loss = output.loss
        if args.enable_rl:
            loss = policy_grad(args, loss, output.logits, tokenizer, batch[TITLE])

        loss /= args.accu_grad
        loss.backward()
        train_loss += loss.item()

        if ((step + 1) % args.accu_grad == 0) or (step + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            if not args.no_scheduler:
                for i in range(args.accu_grad):
                    scheduler.step()

        # Print training loss and accuracy over past logging step
        if (step + 1) % args.logging_step == 0:
            print(f"Epoch {epoch + 1} | Step {step + 1} | loss = {train_loss / args.logging_step:.3f}")
            train_losses.append(train_loss / args.logging_step)
            train_loss = 0.0
    
    return train_losses

def dev_epoch(args, model, dev_loader, tokenizer, device):
    model.eval()
    titles, preds, ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            generate_ids = model.generate(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device), 
                max_length=args.title_max_len,
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

def save_models(args, model, optimizer, scheduler):
    # Save a model and its configuration file to the directory 「saved_model」 
    # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
    # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
    print("Saving Model ...")
    model.save_pretrained(args.ckpt_dir / "model")
    torch.save(optimizer.state_dict(), args.ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), args.ckpt_dir / "scheduler.pt") 

def main(args):
    set_seed(args.seed)
    if args.enable_rl:
        print("Enable reinforcement learning")
        from torch.distributions import Categorical
        from torch.autograd import Variable

    # Data
    SPLITS, filenames = get_splits_n_filenames(args)
    raw_data = {split: read_data(filenames[split], KEYS[split]) for split in SPLITS}
    
    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    
    # Dataset
    dataset = {
        split: NewsSummaryDataset(raw_data[split], tokenizer, args.title_max_len, args.maintext_max_len) 
        for split in SPLITS
        }

    if not args.no_train:
        train_loader = DataLoader(dataset[TRAIN], batch_size=args.batch_size, shuffle=True, pin_memory=True)
        dev_loader = DataLoader(dataset[DEV], batch_size=args.batch_size, shuffle=False, pin_memory=True)

        if args.resume_train:
            args.model_path = args.ckpt_dir / "model"
        elif args.load_model_ckpt is not None:
            args.model_path = args.load_model_ckpt
        # Model
        model = MT5ForConditionalGeneration.from_pretrained(args.model_path).to(device=args.device)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
        # Scheduler
        if args.no_scheduler:
            scheduler = None
        else: 
            total_steps = len(train_loader) * args.num_epoch
            warmup_steps = int(total_steps * args.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= warmup_steps, num_training_steps=total_steps) 
        
        if args.resume_train:
            print(f"Load optimizer and scheduler from {args.ckpt_dir}")
            optimizer.load_state_dict(torch.load(args.ckpt_dir / "optimizer.pt"))
            if not args.no_scheduler:
                scheduler.load_state_dict(torch.load(args.ckpt_dir / "scheduler.pt"))

        train_losses, dev_f = [], []
        best_f = args.resume_best_f
        for epoch in range(args.resume_epoch, args.num_epoch):
            # TRAIN
            new_train_losses = train_epoch(epoch, args, model, train_loader, optimizer, scheduler, args.device)
            train_losses.extend(new_train_losses)
            # DEV
            titles, preds, _ = dev_epoch(args, model, dev_loader, tokenizer, args.device)
            eval_res = get_rouge(preds, titles)
            
            print(eval_res)
            with open(args.log_dir / f"valid_{epoch+1}.json", "w") as fp:
                json.dump(eval_res, fp, indent = 4)
            
            dev_f.append(eval_res["rouge-l"]["f"])
            
            if eval_res["rouge-l"]["f"] >= best_f: 
                best_f = eval_res["rouge-l"]["f"]
                save_models(args, model, optimizer, scheduler)

        if args.make_csv:
            with open(args.log_dir / "loss_f.csv","a+") as fp:
                writer = csv.writer(fp)
                writer.writerow(train_losses)
                writer.writerow(dev_f)
    
    if not args.no_test:
        print("Start Testing...")
        test_loader = DataLoader(dataset[TEST], batch_size=args.batch_size, shuffle=False, pin_memory=True)
        model = MT5ForConditionalGeneration.from_pretrained(args.ckpt_dir / "model").to(device=args.device)
        # TEST
        _, preds, ids = dev_epoch(args, model, test_loader, tokenizer, args.device)
        # Output the prediction
        if args.output_file_path is None:
            args.output_file_path = args.log_dir / f"submission{args.experiment_number}.jsonl"
        with open(args.output_file_path, "w", encoding="utf8") as fp:
            for pred ,i,in zip(preds, ids):
                json.dump({"title":pred, "id":i}, fp, ensure_ascii = False)
                fp.write("\n")
            
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--maintext_max_len", type=int, default=2048)
    parser.add_argument("--title_max_len", type=int, default=64)
    
    parser.add_argument("--model_path", type=str, default="google/mt5-small")
    parser.add_argument("--experiment_number", type=int, default=0)
    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)

    # training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accu_grad", type=int, default=1)
    parser.add_argument("--logging_step", type=int, default=500)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--no_scheduler", action="store_true")


    # decoder
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--early_stopping", action="store_true")

    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    parser.add_argument("--fp16_training", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--make_csv", action="store_true", help="Print loss and rouge score.")

    # Mode
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--resume_train", action="store_true")
    parser.add_argument("--resume_best_f", type=float, default=0.0)
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--load_model_ckpt", type=str, default=None)

    # RL
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--enable_rl", action="store_true")
    

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./log/",
    )
    parser.add_argument(
        "--train_file_path",
        type=Path,
        help="Directory to save the model file.",
        default="./data/train.jsonl",
    )
    parser.add_argument(
        "--dev_file_path",
        type=Path,
        default="./data/public.jsonl",
    )

    # For TA to run
    parser.add_argument(
        "--test_file_path",
        type=Path,
        default="./data/public.jsonl",
    )
    parser.add_argument(
        "--output_file_path",
        type=Path,
        default=None,
    )

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir = args.ckpt_dir / str(args.experiment_number)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = args.log_dir / str(args.experiment_number)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    main(args)
