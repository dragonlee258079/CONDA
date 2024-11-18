#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-359 -p alvis
#SBATCH -t 2-12:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --job-name=CoSOD_B2

python train_3.py
python test_2.py
python evaluation/eval_from_imgs_2.py

