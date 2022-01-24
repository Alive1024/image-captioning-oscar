
DATASET_ROOT="/home/coco_caption/"
MODEL_DIR="inference_models/Oscar"
OUTPUT_DIR="logs/eval_results"

source /etc/profile

python oscar/run_captioning.py \
    --data_dir $DATASET_ROOT \
    --do_test \
    --do_eval \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir $MODEL_DIR
    --output_dir $OUTPUT_DIR
