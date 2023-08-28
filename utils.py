import torch
from torch import _dynamo
import torch._inductor.compile_fx
import ctypes
from typing import List
import traceback
from timer import Timer

_cudart = ctypes.CDLL('libcudart.so')

def profile_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def profile_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)
    
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True
torch._dynamo.config.output_code=True
import logging

num_graph = 0
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    global num_graph
    # logging.info("graph break!")
    # print(gm.graph)
    # print(dir(gm.graph))
    # for node in gm.graph.nodes:
    #     print(node, node.meta)
    # print("example_inputs:", example_inputs)
    num_graph += 1
    return gm.forward


def explain(compiled_func, *args, **kwargs):
    (
        explanation,
        out_guards,
        graphs,
        ops_per_graph,
        break_reasons,
        explanation_verbose,
    ) = torch._dynamo.explain(compiled_func, *args, **kwargs)
    print(explanation_verbose)
    for i, (graph_guard, graph, ops, break_reason) in enumerate(zip(
        out_guards, graphs, ops_per_graph, break_reasons
    )):
        print("GRAPH", i)
        print("++graph_guard:", len(graph_guard))
        for guard in graph_guard:
            print(guard)
        print("++graph:")
        print(graph.print_readable(print_output=False))
        print("++ops:", len(ops))
        for op in ops:
            print(op)
        print("++break_reason:", break_reason.reason)
        print("".join(traceback.format_list(break_reason.user_stack)))
    print("finish")


def perf(repeat=100, sync=True, nvprof=True):
    def wrapper1(func):
        def wrapper(*args, **kwargs):
            for _ in range(repeat):
                o = func(*args, **kwargs)
            if nvprof:
                profile_start()
            if sync:
                torch.cuda.synchronize()
            timer = Timer()
            timer.start()
            for _ in range(repeat):
                o = func(*args, **kwargs)
                if sync:
                    torch.cuda.synchronize()
                timer.log()
            if nvprof:
                profile_stop()
            timer.report()
            return o
        return wrapper
    return wrapper1


def perf_test_run(f, compile_mode, repeat, *args, **kwargs):
    for idx in range(repeat):
        torch.cuda.synchronize()
        o = f(*args, **kwargs)
        torch.cuda.synchronize()
    
    profile_start()
    timer = Timer()
    for idx in range(repeat):
        torch.cuda.synchronize()
        timer.start()
        o = f(*args, **kwargs)
        torch.cuda.synchronize()
        timer.log()
    profile_stop()
    print("compile_mode:", compile_mode)
    timer.report()


def perf_test(f, compile_mode, args, kwargs={}, num_repeat=100):
    repeat = 100
    if compile_mode == "trace":
        # only when kwargs is empty
        if len(kwargs) > 0:
            raise ValueError("kwargs must be empty when compile_mode is trace")
        compiled = torch.jit.trace(f, args, strict=False)
    elif compile_mode == "dynamo":
        torch._dynamo.reset()
        compiled = torch.compile(f)
    elif compile_mode == "dynamo_graph":
        torch._dynamo.reset()
        explain(f, *args, **kwargs)
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=custom_backend)
    elif compile_mode == "eager":
        compiled = f
    elif compile_mode == "fxtrace":
        if len(kwargs) > 0:
            raise ValueError("kwargs must be empty when compile_mode is fxtrace")
        fx_graph = torch.fx.symbolic_trace(f)
        compiled = torch._inductor.compile_fx.compile_fx(fx_graph, args)
    elif compile_mode == "trace+fx": # measure trace time + fx compile time
        if len(kwargs) > 0:
            raise ValueError("kwargs must be empty when compile_mode is trace+fx")
        fx_graph = torch.fx.symbolic_trace(f)
        compiled_func = torch._inductor.compile_fx.compile_fx(fx_graph, args)
        def fn(*args):
            fx_graph = torch.fx.symbolic_trace(f) # intentionally retrace for time measure
            return compiled_func(*args)
        compiled = fn
    else:
        raise NotImplementedError
    global num_graph
    if compile_mode == "dynamo_graph":
        num_graph = 0

    perf_test_run(compiled, compile_mode, repeat, *args)

    if compile_mode == "dynamo_graph":
        print("num_graph:", num_graph)
        num_graph = 0
