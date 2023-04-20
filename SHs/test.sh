#!/bin/bash

#SBATCH --time=40:00:00
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --gres=gpu:a100-20


#__conda_setup="$('/mundus/vgomesma005/ls/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
__conda_setup="$('/mundus/vgomesma005/ls/envs/homographyloss/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"


conda activate rltorch
date

tpath='/mundus/vgomesma005/rl_corrector/datasets/Xu_Hydrofoil_Useed1236_Train117000s.csv'
python ~/rl_corrector/main.py --path=${tpath} --epoch=20 --actor-lr=1e-6 --critic-lr=1e-6 --step-per-epoch=500 --hidden-sizes='10, 10'

date
