#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=tong.su@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.8
# virtualenv --no-download env
source env/bin/activate
module load python/3.8
module load cuda/11.4 gcc/9.3.0 arrow/8.0.0

# pip install .
# pip install azure-cognitiveservices-vision-computervision
# pip install msrest
# pip install Pillow
# pip install pandas
# pip install numpy
# pip install azure-storage-blob
# pip install azure-cognitiveservices-vision-face

python3 image.py
