set -u

DATASET_ROOT="/home/coco_caption/"
S1_PRETRAINED_MODEL_DIR="pretrained_models/base-vg-labels/ep_67_588997"
S1_OUTPUT_DIR="logs/"
  
source /etc/profile

# Training - Step 1: train with cross-entropy loss
python oscar/run_captioning.py \
    --data_dir $DATASET_ROOT \
    --model_name_or_path $S1_PRETRAINED_MODEL_DIR \
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
