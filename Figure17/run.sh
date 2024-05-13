#!/bin/bash
PYTHON_DIR=/home/heheda/envs/frontend-py.sh

TIME_TAG=`date +%y%m%d-%H%M%S`

cd ..
LOG_DIR=logs/dyn-shape
mkdir -p $LOG_DIR

for model in bert deberta densenet resnet
do
    for compile in eager dynamo-dynamic sys-dynamic
    do
            srun -p octave --gres=gpu:1 -J $model-$compile-dynbs /home/heheda/envs/frontend-py.sh run.py --model $model --compile $compile --dyn_bs --repeat=90 2>&1 | tee $LOG_DIR/$model-bs.$compile.log &
    done
done

for model in bert deberta
do
    for compile in eager dynamo-dynamic sys-dynamic
    do
            srun -p octave --gres=gpu:1 -J $model-$compile-dynseq /home/heheda/envs/frontend-py.sh run.py --model $model --compile $compile --dyn_len --repeat=90 2>&1 | tee $LOG_DIR/$model-seqlen.$compile.log &
    done
done

wait