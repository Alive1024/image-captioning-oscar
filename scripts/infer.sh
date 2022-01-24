
DATASET_ROOT="/home/coco_caption/"
MODEL_DIR="logs/checkpoint-2-20000"

python oscar/run_captioning.py \
    --data_dir $DATASET_ROOT \
    --do_test \
    --do_eval \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir $MODEL_DIR
