#!/bin/bash
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --mem=16G
source ./venv/bin/activate
python3 -W ignore word2vec.py