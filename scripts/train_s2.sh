set -u

DATASET_ROOT="/home/coco_caption/"
# Fill in this using one of the log directories of Step'1 output before starting Step 2, such as "logs/checkpoint-0-5000"
S2_PRETRAINED_MODEL_DIR="logs/CIDEr/checkpoint-0-2000"
S2_OUTPUT_DIR="logs/CIDEr"

source /etc/profile

# Training - Step 2: finetune with CIDEr optimization
python oscar/run_captioning.py \
    --data_dir $DATASET_ROOT \
    --model_name_or_path $S2_PRETRAINED_MODEL_DIR \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.000005 \
    --per_gpu_train_batch_size 8 \
    --num_train_epochs 5 \
    --scst \
    --save_steps 2000 \
    --output_dir $S2_OUTPUT_DIR
