#!/bin/bash
set -x
PYTHON_DIR=/home/heheda/envs/frontend-py.sh

TIME_TAG=`date +%y%m%d-%H%M%S`

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export GPU_NUM_DEVICES=1
export PJRT_DEVICE=GPU

mkdir -p logs/$TIME_TAG

bs=1
model=align 
# model=bert
# model=deberta 
# model=densenet
# model=monodepth
# model=quantized
# model=resnet
# model=tridentnet
compile=script
# compile=dynamo
export PYTHONPATH=/home/heheda/frontend/frontend
python3 run.py --bs $bs --model $model --compile $compile 2>&1 | tee logs/$TIME_TAG/$model.$bs.$compile.log
