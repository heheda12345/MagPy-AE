#!/bin/bash

TIME_TAG=`date +%y%m%d-%H%M%S`

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
export GPU_NUM_DEVICES=1
export PJRT_DEVICE=GPU

cd ..
LOG_DIR=logs/e2e-$TIME_TAG
mkdir -p $LOG_DIR

for bs in 1 16
do
    for model in resnet
    do
        for compile in xla dynamo-xla sys-xla
        # for compile in eager dynamo sys script sys-torchscript xla dynamo-xla sys-xla
        do
            if [[ ($model == "align" || $model == "resnet") && ($compile == "xla" || $compile == "dynamo-xla" || $compile == "sys-xla") ]]; then
                echo "export NVIDIA_TF32_OVERRIDE=0 for $model $compile"
                export NVIDIA_TF32_OVERRIDE=0
            fi
            srun -p twills --gres=gpu:h100:1 -J $bs-$model-$compile --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --bs $bs --model $model --compile $compile 2>&1 | tee $LOG_DIR/$model.$bs.$compile.log &
        done
    done
done

wait
