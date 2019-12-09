#!/bin/bash


#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=16g
#SBATCH -p volta-gpu
#SBATCH -t 10-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source activate tf_v100

python ./train_val_generation.py -t ../data/shapenet_partseg/train_val_files.txt -v ../data/shapenet_partseg/test_files.txt -s ../models/generation/ -m pointcnn_seg -x shapenet_generation
