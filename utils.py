import torch
from torch import _dynamo
import torch._inductor.compile_fx
import ctypes
from typing import List
import traceback
from timer import Timer
import numpy as np
from typing import Iterable
from frontend import config as sys_config
from frontend.utils import enable_dyn_shape
from frontend.compile import compile as sys_compile
import logging
import torch_xla.core.xla_model as xm
import torch._dynamo.backends.torchxla

_cudart = ctypes.CDLL('libcudart.so')

def profile_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def profile_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)
    
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.verbose=True
# # torch._dynamo.config.output_code=True
# import logging
# logging.basicConfig(level=logging.INFO)

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
    if torch.__version__ >= "2.1.0":
        torch._logging.set_logs(bytecode=True)
        torch._dynamo.reset()
        explain_output = torch._dynamo.explain(compiled_func)(*args, **kwargs)
        print(explain_output)
        return
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


def assert_equal(ref, out):
    precision = 5e-3
    assert type(ref) == type(
        out), f"wrong type: expect {type(ref)}, got {type(out)}"
    if isinstance(ref, torch.Tensor):
        assert (isinstance(out, torch.Tensor))
        r = ref.cpu()
        o = out.cpu()
        if r.dtype == torch.bool and o.dtype == torch.int8:
            o = o.bool()
        all_close = torch.allclose(r, o, atol=precision, rtol=precision)
        if not all_close:
            close = torch.isclose(r, o, rtol=precision, atol=precision)
            print("ref:", torch.masked_select(r, ~close))
            print("out:", torch.masked_select(o, ~close))
            print(torch.sum(~close))
            print("wrong answer !!!!!!!!!!!!!!!!!!!!!!!!!!")
            assert (False)
    elif isinstance(ref, Iterable):
        assert (isinstance(out, Iterable))
        if isinstance(ref, dict):
            assert (len(ref) == len(out))
            for k, v in ref.items():
                assert_equal(v, out[k])
        else:
            for r, o in zip(ref, out):
                assert_equal(r, o)
    else:
        assert ref == out, f"wrong answer: expect {ref}, got {out}"


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


def perf_test_run(f, compile_mode, repeat, args, kwargs):
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


def perf_test_run_cf(f, compile_mode, repeat, args_all, kwargs_all):
    for idx in range(repeat):
        print("warmup:", idx)
        torch.cuda.synchronize()
        o = f(*args_all[idx], **kwargs_all[idx])
        torch.cuda.synchronize()
    
    profile_start()
    timer = Timer()
    for idx in range(repeat):
        print("run:", idx)
        torch.cuda.synchronize()
        timer.start()
        o = f(*args_all[idx], **kwargs_all[idx])
        torch.cuda.synchronize()
        timer.log()
    profile_stop()
    print("compile_mode:", compile_mode)
    timer.report()

def perf_test_run_bs(orignal, f, compile_mode, num_repeat, get_input_fn):
    bs_list = list(range(2, 17))
    assert num_repeat % len(bs_list) == 0
    num_repeat_per_bs = num_repeat // len(bs_list)
    # compile with bs=5 to avoid specialization
    args, kwargs = get_input_fn(5)
    o = f(*args, **kwargs)

    for i in range(num_repeat_per_bs):
        for bs in bs_list:
            args, kwargs = get_input_fn(bs)
            if i == 0:
                expect = orignal(*args, **kwargs)
            torch.cuda.synchronize()
            o = f(*args, **kwargs)
            torch.cuda.synchronize()
            if i == 0:
                assert_equal(expect, o)
    
    profile_start()
    timer = Timer()
    for i in range(num_repeat_per_bs):
        for bs in bs_list:
            # print("run:", i, bs, flush=True)
            args, kwargs = get_input_fn(bs)
            torch.cuda.synchronize()
            timer.start()
            o = f(*args, **kwargs)
            torch.cuda.synchronize()
            timer.log()
    profile_stop()
    print("compile_mode:", compile_mode)
    timer.report()

def perf_test_run_seq_len(orignal, f, compile_mode, num_repeat, get_input_fn):
    len_list = list([x * 16 for x in range(2, 17)])
    assert num_repeat % len(len_list) == 0
    num_repeat_per_bs = num_repeat // len(len_list)
    # compile with bs=5 to avoid specialization
    batch_size = 8
    args, kwargs = get_input_fn(batch_size, 80)
    o = f(*args, **kwargs)
    # print("end!!!!!!!!!!!!!!!!!")
    # exit(0)

    for i in range(num_repeat_per_bs):
        for seq_len in len_list:
            args, kwargs = get_input_fn(batch_size, seq_len)
            if i == 0:
                expect = orignal(*args, **kwargs)
            torch.cuda.synchronize()
            o = f(*args, **kwargs)
            torch.cuda.synchronize()
            if i == 0:
                assert_equal(expect, o)
    
    profile_start()
    timer = Timer()
    for i in range(num_repeat_per_bs):
        for seq_len in len_list:
            # print("run:", i, bs, flush=True)
            args, kwargs = get_input_fn(batch_size, seq_len)
            torch.cuda.synchronize()
            timer.start()
            o = f(*args, **kwargs)
            torch.cuda.synchronize()
            timer.log()
    profile_stop()
    print("compile_mode:", compile_mode)
    timer.report()


# import torch._dynamo.config
# import logging
# torch._dynamo.config.verbose=True
# torch._dynamo.config.output_code=True

def perf_test(f, compile_mode, args, kwargs, get_input_fn, num_repeat, dynamic_mode, check):
    # logging.basicConfig(level=logging.INFO, force=True)
    if compile_mode == "trace":
        # only when kwargs is empty
        if len(kwargs) > 0:
            raise ValueError("kwargs must be empty when compile_mode is trace")
        compiled = torch.jit.trace(f, args, strict=False)
    elif compile_mode == "dynamo":
        torch._dynamo.reset()
        compiled = torch.compile(f)
    elif compile_mode == "dynamo-tensorrt":
        import torch_tensorrt
        torch._dynamo.reset()
        compiled = torch_tensorrt.dynamo.compile(f, *args, **kwargs)
    elif compile_mode == "dynamo-xla":
        torch._dynamo.reset()
        args = tuple((arg.to('cpu').to(xm.xla_device()) for arg in args))
        kwargs = dict({k: v.to('cpu').to(xm.xla_device()) for k, v in kwargs.items()})
        compiled_f = torch.compile(f, backend='aot_torchxla_trace_once')
        def f_with_sync(*args, **kwargs):
            o = compiled_f(*args, **kwargs)
            xm.mark_step()
            xm.wait_device_ops()
            return o
        compiled = f_with_sync
    elif compile_mode == "dynamo-dynamic":
        torch._dynamo.reset()
        compiled = torch.compile(f, dynamic=True)
    elif compile_mode == "dynamo_graph":
        torch._dynamo.reset()
        if dynamic_input:
            explain(f, *args[0], **kwargs[0])
        else:
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
    elif compile_mode == "sys":
        sys_config.set_config('debug', False)
        compiled = sys_compile(f)
    elif compile_mode == "sys-dynamic":
        sys_config.set_config('debug', False)
        compiled = sys_compile(f)
    elif compile_mode == "sys-xla":
        sys_config.set_config('debug', False)
        sys_config.set_config('backend', 'xla')
        args = tuple((arg.to('cpu').to(xm.xla_device()) for arg in args))
        kwargs = dict({k: v.to('cpu').to(xm.xla_device()) for k, v in kwargs.items()})
        compiled_f = sys_compile(f)
        def f_with_sync(*args, **kwargs):
            o = compiled_f(*args, **kwargs)
            xm.mark_step()
            xm.wait_device_ops()
            return o
        compiled = f_with_sync
    elif compile_mode == 'xla':
        args = tuple((arg.to('cpu').to(xm.xla_device()) for arg in args))
        kwargs = dict({k: v.to('cpu').to(xm.xla_device()) for k, v in kwargs.items()})
        def f_with_sync(*args, **kwargs):
            o = f(*args, **kwargs)
            xm.mark_step()
            xm.wait_device_ops()
            return o
        compiled = f_with_sync
    else:
        raise NotImplementedError
    global num_graph
    if compile_mode == "dynamo_graph":
        num_graph = 0
    if not check: f = compiled
    if dynamic_mode == 'cf':
        perf_test_run_cf(f, compiled, compile_mode, num_repeat, args, kwargs)
    elif dynamic_mode == 'bs':
        if compile_mode == 'sys-dynamic':
            with enable_dyn_shape():
                perf_test_run_bs(f, compiled, compile_mode, num_repeat, get_input_fn)
        else:
            perf_test_run_bs(f, compiled, compile_mode, num_repeat, get_input_fn)
    elif dynamic_mode == 'len':
        if compile_mode == 'sys-dynamic':
            with enable_dyn_shape():
                perf_test_run_seq_len(f, compiled, compile_mode, num_repeat, get_input_fn)
        else:
            perf_test_run_seq_len(f, compiled, compile_mode, num_repeat, get_input_fn)
    else:
        perf_test_run(compiled, compile_mode, num_repeat, args, kwargs)


    if compile_mode == "dynamo_graph":
        print("num_graph:", num_graph)
        num_graph = 0


def read_bin(s, dtype=np.float32):
    with open(s + ".shape") as f: shape = tuple((int(x) for x in f.read().strip().split(" ")))
    tensor = torch.from_numpy(np.fromfile(s + ".bin", dtype=dtype)).reshape(shape)
    return tensor


def save_bin(data, path):
    data = data.clone().detach().cpu().numpy()
    with open(path + ".shape", "w") as f: f.write(" ".join(str(x) for x in data.shape))
    data.tofile(path + ".bin")
