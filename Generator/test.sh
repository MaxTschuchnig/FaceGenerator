#!/bin/bash

echo $(date)

export PATH=~/miniconda/bin/:$PATH
source activate MasterenvGPU

CUDA_VISIBLE_DEVICES=1 python Faces.py '1' 0.0001 0.5 0.001 0.5 0.2 0.02
