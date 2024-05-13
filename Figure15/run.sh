#!/bin/bash
PYTHON_DIR=/home/heheda/envs/frontend-py.sh

TIME_TAG=`date +%y%m%d-%H%M%S`

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export GPU_NUM_DEVICES=1
export PJRT_DEVICE=GPU

cd ..
LOG_DIR=logs/e2e
mkdir -p $LOG_DIR

for bs in 1 16
do
    for model in align bert deberta densenet monodepth quantized resnet tridentnet
    do
        for compile in eager dynamo sys script sys-torchscript xla dynamo-xla sys-xla
        do
                srun -p octave --gres=gpu:1 -J $bs-$model-$compile /home/heheda/envs/frontend-py.sh run.py --bs $bs --model $model --compile $compile 2>&1 | tee $LOG_DIR/$model.$bs.$compile.log &
        done
    done
done

wait
