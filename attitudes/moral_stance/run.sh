#!/bin/bash
#SBATCH --partition=donut-default
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50GB

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/anaconda3/lib

echo "MODEL: xlmt, TRAIN: EN MF tweets + 5k incas, TEST: 5k incas"
python finetune_transformer.py --mode train_and_test --train_path ~/ace_elect/en_mf_train.csv --test_path ~/ace_elect/incas_test.csv -o ./model_outputs_xlmt -m "cardiffnlp/twitter-xlm-roberta-base" -e 15
