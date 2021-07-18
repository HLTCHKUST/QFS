# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
# you need to specify data_dir, output_dir and model_name_or_path
export DATA_DIR=/path/to/data/dir/
export OUTPUT_DIR=/path/to/your/output/dir/

nohup python3 train_qfs.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path facebook/bart-large-xsum \
    --learning_rate=3e-5 \
    --gpus 1 \
    --fp16 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --max_source_length 142 \
    --max_target_length 48 \
    --val_max_target_length 48 \
    --test_max_target_length 48 \
    --freeze_embeds \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_workers 8 \
    --gradient_accumulation_steps 4 \
    > log.txt &
    $@ 
