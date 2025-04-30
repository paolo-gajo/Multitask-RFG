#!/bin/bash
#SBATCH -J run
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

arc_norm_opts=(0 1)
seed_opts=(0 1 2 3 4)

for seed in "${seed_opts[@]}"
do
    for arc_norm in "${arc_norm_opts[@]}"
    do
        python ./tools/train.py --opts \
        --arc_norm $arc_norm \
        --seed $seed
    done
done
