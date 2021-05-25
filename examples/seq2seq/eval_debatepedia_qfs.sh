# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
DEBATEPEDIA_DIR=/home/qasystem/sudan/bart/data/Debatepedia/data_fold_1
# DEBATEPEDIA_DIR=/home/sudan/Kaggle/bart/data/Debatepedia/data_fold_1
# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path

export DATA_DIR=/home/qasystem/sudan/bart/data/Debatepedia/data_fold_1
nohup python3 run_eval_qfs.py --model_name ./debatepedia_results_qfs/best_tfmr \
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
