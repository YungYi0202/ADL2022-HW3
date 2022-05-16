# ADL2022 HW3 Readme

## Reproduce the prediction

```
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```
## How to train my model

Please first install tw_rouge. [Link](https://github.com/moooooser999/ADL22-HW3)

### Parameters
- CKPT_DIR: path/to/dir/of/ckpt
- LOG_DIR: path/to/dir/of/log
- EXPRIMENT_NUM, QA_EXPRIMENT_NUM
    - To distinguish different attempt.
    - The checkpoint files would be stored in `CKPT_DIR / EXPRIMENT_NUM /`
    - The rouge score of each epoch would be stored in files `valid_{epoch}.json` in `LOG_DIR / EXPRIMENT_NUM /`.

### Reproduce the model I submitted.
```
python3.8 train.py \
    --experiment_number EXPRIMENT_NUM \
    --ckpt_dir CKPT_DIR \
    --log_dir LOG_DIR \
    --batch_size 1 \
    --accu_grad 8 \
    --num_epoch 25 \
    --num_beams 10 \
    --lr 5e-5 \
```

### Bonus: Reproduce the result of RL
```
python3.8 train.py \
    --experiment_number EXPRIMENT_NUM \
    --ckpt_dir CKPT_DIR \
    --log_dir LOG_DIR \
    --batch_size 1 \
    --accu_grad 8 \
    --num_epoch 10 \
    --num_beams 10  \
    --make_csv \
    --lr 5e-5 \
    --enable_rl \
    --load_model_ckpt ./hw3_model/ \
```
