# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
# you need to specify data_dir, output_dir and model_name_or_path
export OUTPUT_DIR=/path/to/your/output/dir/
export DATA_DIR=/path/to/data/dir/

nohup python3 run_eval_qfs.py --model_name $OUTPUT_DIR/best_tfmr \
    --input_dir $DATA_DIR \
    --save_path debatepedia_test_generations.txt \
    --reference_path $DATA_DIR/test.target \
    --score_path debatepedia_rouge.json \
    --task summarization \
    --n_obs 100 \
    --device cuda \
    --fp16 \
    --bs 32 \
    > log.txt &
    $@ 
