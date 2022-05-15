python3.8 inference.py \
    --test_file_path ${1} \
    --output_file_path ${2} \
    --maintext_max_len 1536 \
    --load_model_ckpt ./hw3_model/ \
    --num_beams 10 \