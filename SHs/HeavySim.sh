#!/bin/bash

#SBATCH --time=40:00:00
#SBATCH --job-name=YAWPre20tan
#SBATCH --output=YAWPre20tan.out
#SBATCH --error=YAWPre20tan.err
#SBATCH --gres=gpu:a100-20


#__conda_setup="$('/mundus/vgomesma005/ls/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
__conda_setup="$('/mundus/vgomesma005/ls/envs/homographyloss/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"


conda activate rltorch
date

tpath='/mundus/vgomesma005/Identifier20Steps/Xu_Identification_Useed1236_Train144000s.csv'
python ~/rl_corrector/main.py --path=${tpath} --epochs=20 --actor-lr=1e-6 --critic-lr=1e-6 --step-per-epoch=500 --hidden-sizes=[10, 128]

date
