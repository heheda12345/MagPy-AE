# Environment Setup
This experiment uses Cocktailer(OSDI'23) as the backend compiler to compie the control flow operator. In this guide, we will show how to setup the environment of Cocktailer.

** For AE Reviewers **: We have prepared the environment in our cluster. You can skip this section and directly use the run.sh as step 3 to reproduce the results.

1. Install Cocktailer
```bash
cd $YOUR_DIR_FOR_NNFUSION
git clone https://github.com/microsoft/nnfusion.git --branch cocktailer_artifact --single-branch

# prepare the environment (needs sudo)
bash nnfusion/maint/script/install_dependency.sh

# build the C++ parts
cd $YOUR_DIR_FOR_NNFUSION/nnfusion
git apply $AE_DIR/Figure18/nnfusion.patch
mkdir build && cd build && cmake .. && make -j
export PATH=$YOUR_DIR_FOR_NNFUSION/nnfusion/build/src/tools/nnfusion:$PATH

# build the Python parts
cd $YOUR_DIR_FOR_NNFUSION/nnfusion/src/python
pip install -e .
```

2. Install fx2onnx
Cocktailer needs operator graph in ONNX format, so we build fx2onnx to convert the torch.fx graph of DeepVisor to ONNX format.
```bash
cd $AE_DIR/fx2onnx
pip install -e .
```

3. Reproduce the result
You can use the `run.sh` in this folder to reproduce the result
```bash
./run.sh
```
