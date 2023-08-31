#!/bin/bash
ROOT=$PWD/../..
cd $ROOT
git clone git@git.tsinghua.edu.cn:frontend/models/transformers.git --branch exp
cd transformers && pip install -e . && cd ..
