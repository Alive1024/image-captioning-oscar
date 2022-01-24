set -u

DATASET_ROOT="/home/coco_caption/"
S1_OUTPUT_DIR="logs/"
S2_OUTPUT_DIR="logs/CIDEr"

# Step 1: train with cross-entropy loss
python oscar/run_captioning.py \
    --data_dir $DATASET_ROOT \
    --model_name_or_path pretrained_models/base-vg-labels/ep_67_588997 \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.00003 \
    --per_gpu_train_batch_size 64 \
    --num_workers 32 \
    --num_train_epochs 30 \
    --save_steps 5000 \
    --output_dir $S1_OUTPUT_DIR

# Step 2: finetune with CIDEr optimization
python oscar/run_captioning.py \
    --data_dir $DATASET_ROOT \
    --model_name_or_path $S1_OUTPUT_DIR \
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
