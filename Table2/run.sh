#!/bin/bash
PYTHON_DIR=/home/heheda/envs/frontend-py.sh

TIME_TAG=`date +%y%m%d-%H%M%S`

cd ..
LOG_DIR=logs/graphcount
mkdir -p $LOG_DIR

for bs in 1
do
    for model in align bert deberta densenet monodepth quantized resnet tridentnet
    do
        for compile in dynamo-graph script
        do
                srun -p octave --gres=gpu:1 -J graphcount-$model-$compile /home/heheda/envs/frontend-py.sh run.py --bs $bs --model $model --compile $compile 2>&1 | tee $LOG_DIR/$model.$compile.log &
        done
    done
done

wait

cd -

python3 gather.py # print all graph count