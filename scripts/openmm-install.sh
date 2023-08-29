#!/bin/bash
ROOT=$PWD/../..
cd $ROOT
# install mmcv
git clone git@github.com:open-mmlab/mmcv.git --branch v1.7.1
cd mmcv && MMCV_WITH_OPS=1 pip install -e . && cd ..
# install mmdet
git clone git@git.tsinghua.edu.cn:frontend/models/mmdetection.git --branch exp
cd mmdetection && pip install -e . && cd ..