# ATC'24 MagPy Artifact

## 1. Overview
This repository contains the artifact for the paper "MagPy: Effective Operator Graph Instantiation for Deep Learning by Execution State Monitoring" to be appeared in USENIX ATC'24. MagPy is a JIT compiler for PyTorch programs. It can extract the operator graph from PyTorch programs and optimize the graph with a wide range of deep learning graph compilers.

### Evaluation Setup
* Artifacts Available:
    * All MagPy related code are available under [https://github.com/heheda12345/MagPy](https://github.com/heheda12345/MagPy), and all artifact scripts are available under [https://github.com/heheda12345/MagPy-AE](https://github.com/heheda12345/MagPy-AE)
* Artifacts Functional:
    * *Documentation*: the following of documents include detailed guidelines on how to build, install, test MagPy and the experiments to compare with other baselines.
    * *Completeness*: the source code of MagPy is located at ``MagPy/frontend``, with `cache.py` as the cache, `object_table.py` as the RefGraph, `variables` folder as different type of ShadowNodes and ShadowVersion. 
    * *Exercisability*: in this repository, we prepare all the script and data to reproduce the experiements in individual folders named by the figure or table id in paper.
* Results Reproduced:
    * To reproduce the main results presented in our paper, we provide detailed guideline to install the softwares with the same configurations as we used in paper evauation, and scripts to help reproduce the results step by step.


## 2. Environment Preparation

* Download code
    ```bash
    git clone https://github.com/heheda12345/MagPy.git
    git clone https://github.com/heheda12345/MagPy-AE.git
    export FRONTEND_DIR=$PWD/MagPy
    export AE_DIR=$PWD/MagPy-AE
    ```

* Prerequisite
    * Python 3.9 (Other Python version is not supported yet)
    * CUDA 11.8 (Fully tested, other versions may work)

* Install dependencies
The README.md in frontend repo only provides the guide to run MagPy. More dependencies are needed to reproduce the results in this paper. Please use the following steps to step up the environment.

    ```bash
    cd $AE_DIR && pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html && cd ..
    cd $FRONTEND_DIR && pip install -e . && cd ..
    sed -i 's/py_all(a\.shape\[i\] for i in dims)/py_all(a.shape[i] > 0 for i in dims)/g' `python3 -c 'import torch; print(torch.__path__[0])'`/_refs/__init__.py
    sed -i "226,230s/^/# /"  `python3 -c 'import torch_xla; print(torch_xla.__path__[0])'`/core/dynamo_bridge.py
    ```

    The last two commands are for fixing the following issues:
    1. change the `py_all(a.shape[i] > 0 for i in dims)` to `py_all(a.shape[i] > 0 for i in dims)` in `torch/_refs/__init__.py` to avoid the guard on the reduction length
    2. comment out the following lines of torch_xla to avoid crash
        ```python
        # fallback_ops = get_fallback_ops()
        # if len(fallback_ops) > 0:
        # raise RuntimeError(
        #     f"Fail to extact the compiled graph because of fallback: {','.join(fallback_ops)}"
        # )
        ```

## 3. Getting Started with a Simple Example
* Go to the `get_started_tutorial/` folder and follow [README_GET_STARTED.md](get_started_tutorial/README.md).

## 4. Reproducing Experiement Results

TABLE 3 and Figure 18 needs additional environment setup. Please follow the instructions in [Figure18/README.md](Figure18/README.md) and [Table3/README.md](Table3/README.md) to setup the environment. Then, you can ``cd`` to each folder and use ``./run.sh'' to reproduce the results.

| Experiments   | Figure # in Paper |  Script Location |
| -----------     | -----------  |  ----------- |
| #1. Power of graph instantiation | Figure 1 | N/A (use the results in Figure 15) |
| #2. End-to-end inference on NVIDIA A100 | Figure 15 | [run.sh](Figure15/run.sh) |
| #3. Time breakdown of execution (BS=1) | Figure 16 | [run.sh](Figure16/run.sh) |
| #4. Inference of models with dynamic shape | Figure 17 | [run.sh](Figure17/run.sh) |
| #5. Inference of models with dynamic control flow | Figure 18 | [run.sh](Figure18/run.sh) |
| #6. Number of exported operator graphs | Table 2 | [run.sh](Table2/run.sh) |
| #7. Result on ParityBench | Table 3 | [README.md](Table3/README.md) |

## 5. Reproduce the Figures
You can use the scripts in ``plot`` folder to reproduce the figures. The figures will be saved to `plot/figures/` folder.

```bash
cd $AE_DIR/plot
./plot.sh
```