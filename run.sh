python3.8 inference.py \
    --test_file_path ${1} \
    --output_file_path ${2} \
    --load_model_ckpt ./hw3_model/ \
    --num_beams 5 \
    --batch_size 2 \