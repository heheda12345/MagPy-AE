#!/bin/bash

TIME_TAG=`date +%y%m%d-%H%M%S`

cd ..
LOG_DIR=logs/dyn-shape
mkdir -p $LOG_DIR

for model in bert deberta densenet resnet
do
    for compile in eager dynamo-dynamic sys-dynamic
    do
            srun -p octave --gres=gpu:1 -J $model-$compile-dynbs --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --model $model --compile $compile --dyn_bs --repeat=90 2>&1 | tee $LOG_DIR/$model-bs.$compile.log &
    done
done

for model in bert deberta
do
    for compile in eager dynamo-dynamic sys-dynamic
    do
            srun -p octave --gres=gpu:1 -J $model-$compile-dynseq --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --model $model --compile $compile --dyn_len --repeat=90 2>&1 | tee $LOG_DIR/$model-seqlen.$compile.log &
    done
done

wait