# Get Start Tutorial: Compile a Toy model with MagPy
We assume you already build and install MagPy following the *Installation* section in [README.md](../README.md).

This tutorial will first show how to use MagPy to compile and optimize a simple PyTorch program. Then, we will demonstrate the performance improvement with MagPy compiler for compiling complex PyTorch programs.

*** for AE Reviewers ***
Please use the following command to load the environment in our cluster.
```bash
source ~/ae_env.sh
```

## Run a Simple Program

```bash
cd $AE_DIR/get_started_tutorial
LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 example.py
```

You can find the following contents in the log:
1. The exported graph in torch.fx format
    ```
    graph graph():
        %tensor_1 : [#users=1] = placeholder[target=tensor_1]
        %conv : [#users=1] = call_module[target=conv](args = (%tensor_1,), kwargs = {})
        %relu : [#users=1] = call_module[target=relu](args = (%conv,), kwargs = {})
        return (relu,)
    ```

2. The log from TorchInductor showing it is compiling the graph
    ```
    [2024-05-14 11:39:40,041] torch._inductor.compile_fx: [INFO] Step 1: torchinductor compiling FORWARDS graph 0
    [2024-05-14 11:39:43,936] torch._inductor.compile_fx: [INFO] Step 1: torchinductor done compiling FORWARDS graph 0
    ```

3. The mock code
    ```
    def fn(locals):
        print('running graph_fn (key = 5840)', locals.keys())
        graph_out = compiled_graph(locals['x'].contiguous())
        __stack__0 = graph_out[0]
        return __stack__0
    return fn
    ```


4. The guard
    ```
    def fn(locals):
        try:
            print('running guard_fn (key = 5840)', locals.keys())
            ok = True
            missed_check = []
            if not (id(locals['self']) == 22960477248864):
                missed_check.append((r"locals['self']", r"id(locals['self']) == 22960477248864"))
                ok = False
            if not (obj_2.tensor_guard_check(locals['x'])):
                missed_check.append((r"locals['x']", r"obj_2.tensor_guard_check(locals['x'])"))
                ok = False
            print('ok = ', ok)
        except Exception as e:
            print('exception in guard_fn:', e, type(e))
            import traceback
            print(traceback.format_exc())
            return (missed_check, False)
        return (missed_check, ok)
    ```

5. The log about successful guard match in the second run
    ```
    running guard_fn (key = 5840) dict_keys(['self', 'x'])
    ok =  True
    INFO [example.py:13] guard cache hit: frame_id 0 callsite_id 0
    running graph_fn (key = 5840) dict_keys(['self', 'x', '__case_idx', '__graph_fn'])
    ```

## Run a Complex Program

The following commands compiles the DenseNet model with different approaches. A-B means using A to extract the graph and B to compile the graph.

```bash
cd $AE_DIR

# Eager
srun -p octave --gres=gpu:1 --pty python3 run.py --bs 1 --model densenet --compile eager

# TorchDynamo-Inductor
srun -p octave --gres=gpu:1 --pty python3 run.py --bs 1 --model densenet --compile dynamo

# MagPy-Inductor
srun -p octave --gres=gpu:1 --pty --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --bs 1 --model densenet --compile sys

# TorchScript-TorchScript
srun -p octave --gres=gpu:1 --pty python3 run.py --bs 1 --model densenet --compile script

# MagPy-TorchScript
srun -p octave --gres=gpu:1 --pty --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --bs 1 --model densenet --compile sys-torchscript

# LazyTensor-XLA
XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME GPU_NUM_DEVICES=1 PJRT_DEVICE=GPU srun -p octave --gres=gpu:1 --pty python3 run.py --bs 1 --model densenet --compile xla

# TorchDynamo-XLA
XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME GPU_NUM_DEVICES=1 PJRT_DEVICE=GPU srun -p octave --gres=gpu:1 --pty python3 run.py --bs 1 --model densenet --compile dynamo-xla

# MagPy-XLA
XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME GPU_NUM_DEVICES=1 PJRT_DEVICE=GPU srun -p octave --gres=gpu:1 --pty --export=ALL,LD_PRELOAD=$FRONTEND_DIR/build/ldlong.v3.9.12.so python3 run.py --bs 1 --model densenet --compile sys-xla

```

You can also change --compile as following:

|    --compile    	|     Name in paper     	|
|:---------------:	|:---------------------:	|
| sys-torchscript 	| MagPy-TorchScript 	|
|    dynamo-xla   	|    TorchDynamo-XLA    	|
|     sys-xla     	|     MagPy-XLA     	|

The output of each command should end up with the following log, showing that MagPy can outperform other approaches when using the same graph compiler:
```
compile_mode: eager
100 iters, min = 0.0114 s, max = 0.0123 s, avg = 0.0115 s

----------------------------------------------------------

compile_mode: dynamo
100 iters, min = 0.0239 s, max = 0.0268 s, avg = 0.0242 s

compile_mode: sys
100 iters, min = 0.0126 s, max = 0.0191 s, avg = 0.0128 s

----------------------------------------------------------

compile_mode: script
100 iters, min = 0.0112 s, max = 0.0152 s, avg = 0.0114 s

compile_mode: sys-torchscript
100 iters, min = 0.0059 s, max = 0.0063 s, avg = 0.0060 s

----------------------------------------------------------

compile_mode: xla
100 iters, min = 0.0140 s, max = 0.0151 s, avg = 0.0146 s

compile_mode: dynamo-xla
100 iters, min = 0.0364 s, max = 0.0436 s, avg = 0.0373 s

compile_mode: sys-xla
100 iters, min = 0.0149 s, max = 0.0247 s, avg = 0.0157 s
```
