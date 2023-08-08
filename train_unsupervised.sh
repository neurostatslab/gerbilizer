#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=a100
#SBATCH -c 6
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=6-0
#SBATCH -o slurm_logs/train_unsupervised.log
pwd; hostname; date;

config_path='/mnt/home/atanelus/repos/gerbilizer/unsupervised_config.json5'
output_dir='/mnt/home/atanelus/ceph/unsupervised_model'

data_dir='/mnt/home/atanelus/ceph/preprocess_c5/processed_datasets/'
tmp_dir='/tmp/dataset'
mkdir -p $tmp_dir

echo "Copying data"
cp -r $data_dir/*.h5 $tmp_dir
echo "Done copying data, starting training"

source ~/.bashrc
source /mnt/home/atanelus/venvs/unsupervised/bin/activate

python -u -m gerbilizer.unsupervised \
    --config $config_path \
    --data $tmp_dir \
    --save-path $output_dir \

date;
