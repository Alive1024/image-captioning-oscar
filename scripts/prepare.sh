set -u

DATASET_MOUNT_DIR="/dataset/ac0c36fd/v2"    # The directory where the dataset is mounted on

# Unzip the CoCo Caption Dataset
unzip $DATASET_MOUNT_DIR/coco_caption.zip -d /home/
cp $DATASET_MOUNT_DIR/coco-train-words.p /home/coco_caption


# Install JDK1.8
cd /workspace/oscar_dependencies/
cp jdk1.8.0_311 /usr/local/jdk1.8
echo "export JAVA_HOME=/usr/local/jdk1.8">>/etc/profile
echo "export JRE_HOME=\${JAVA_HOME}/jre">>/etc/profile
echo "export CLASSPATH=.:\${JAVA_HOME}/lib:\${JRE_HOME}/lib">>/etc/profile
echo "export PATH=.:\${JAVA_HOME}/bin:\$PATH">>/etc/profile
/bin/bash -c "source /etc/profile"

# Install apex
cd /workspace/oscar_dependencies/apex
python setup.py install --cuda_ext --cpp_ext

# Install py-bottom-up-attention (detectron2)
cd /workspace/oscar_dependencies/py-bottom-up-attention
python setup.py build develop

# Install other Python dependecies
cd /workspace
pip install -r requirements.txt
