#!/bin/bash
PYTHON_DIR=/home/heheda/envs/frontend-py.sh

TIME_TAG=`date +%y%m%d-%H%M%S`

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export GPU_NUM_DEVICES=1
export PJRT_DEVICE=GPU

mkdir -p logs/$TIME_TAG

for bs in 1 16
do
    for model in align bart deberta densenet monodepth quantized tridentnet
    do
        for compile in eager dynamo sys xla dynamo-xla sys-xla
        do
                srun --exclusive -p Big --gres=gpu:v132p:1 /home/heheda/envs/frontend-py.sh run.py --bs $bs --model $model --compile $compile 2>&1 | tee logs/$TIME_TAG/$model.$bs.$compile.log
        done
    done
done
