set -u

DATASET_MOUNT_DIR="/dataset/ac0c36fd/v2"    # The directory where the dataset is mounted on

# Unzip the CoCo Caption Dataset
unzip $DATASET_MOUNT_DIR/coco_caption.zip -d /home/
cp $DATASET_MOUNT_DIR/coco-train-words.p /home/coco_caption
