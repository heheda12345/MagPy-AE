#!/bin/bash


cd ..
LOG_DIR=logs/control_flow
mkdir -p $LOG_DIR

for bs in 1 16
do
    for model in lstm blockdrop
    do
        for compile in eager sys-nnf
        do
            srun -p octave --gres=gpu:1 -J cf-$bs-$model-$compile --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --bs $bs --model $model --compile $compile --dyn_cf 2>&1 | tee $LOG_DIR/$model.$bs.$compile.log &
        done
    done
    srun -p octave --gres=gpu:1 -J $bs-blockdrop-dynamo-nnf --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --bs $bs --model blockdrop --compile dynamo-nnf --dyn_cf 2>&1 | tee $LOG_DIR/blockdrop.$bs.dynamo-nnf.log &
    srun -p octave --gres=gpu:1 --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 models/lstm.py --bs $bs 2>&1 | tee $LOG_DIR/lstm.$bs.dynamo-nnf.log &
done

wait