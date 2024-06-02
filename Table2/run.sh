#!/bin/bash
TIME_TAG=`date +%y%m%d-%H%M%S`

cd ..
LOG_DIR=logs/graphcount
mkdir -p $LOG_DIR

for bs in 1
do
    for model in align bert deberta densenet monodepth quantized resnet tridentnet
    do
        for compile in dynamo-graph script xla-graph
        do
                srun -p octave --gres=gpu:1 -J graphcount-$model-$compile --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --bs $bs --model $model --compile $compile --repeat 3 2>&1 | tee $LOG_DIR/$model.$compile.log &
        done
    done
done

wait
