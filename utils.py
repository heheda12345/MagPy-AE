import torch
from torch import _dynamo
import torch._inductor.compile_fx
import ctypes
from typing import List
import traceback
from timer import Timer
import time
import numpy as np
from typing import Iterable
from frontend import config as sys_config
from frontend.utils import enable_dyn_shape
from frontend.compile import compile as sys_compile
import logging
import os
import sys
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

def get_inductor_with_profile(timer: Timer):
    import torch._inductor
    def inductor_with_profile(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        start_time = time.time()
        compiled = torch._inductor.compile_fx.compile_fx(gm, example_inputs)
        end_time = time.time()

        def run(*args):
            torch.cuda.synchronize()
            timer.start()
            o = compiled(*args)
            torch.cuda.synchronize()
            timer.log()
            return o
        print(f"compile time: {end_time - start_time} s")
        return run
    return inductor_with_profile


def onnx_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    global num_graph
    real_inputs = tuple([torch.rand(x.shape, dtype=x.dtype, layout=x.layout, device=x.device) for x in  example_inputs])
    input_names = tuple([f"input_{i}" for i in range(len(real_inputs))])
    model_path = f"tmp/onnx_graph_{num_graph}.onnx"
    import onnx
    import onnxruntime as ort
    def load_model(model_path):
        onnx_model = onnx.load(model_path)
        print(onnx.helper.printable_graph(onnx_model.graph))
        # print(onnx_model.graph.value_info)
        onnx.checker.check_model(onnx_model)
        print(f"{model_path}: check passed!")
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        session = ort.InferenceSession(model_path)

        inputs_name = [item.name for item in onnx_model.graph.input]
        outputs_name = [item.name for item in onnx_model.graph.output]
        return session, inputs_name, outputs_name
    torch.onnx.export(gm, real_inputs, model_path, verbose=True, opset_version=12, input_names=input_names, training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False)
    session, onnx_input_names, outputs_name = load_model(model_path)
    def fn(*args):
        ort_inputs = {
            onnx_input_names[i]: args[i].contiguous().cpu().detach().numpy() for i in range(len(args))
        }
        ort_outputs = session.run(outputs_name, ort_inputs)
        output_gm = list(gm.forward(*args))
        output_ort = list([torch.from_numpy(item).cuda() for item in ort_outputs])
        assert_equal(output_gm, output_ort)
        return output_ort
    num_graph += 1
    return fn


def nnf_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    global num_graph
    num_graph += 1
    real_inputs = tuple([torch.rand(x.shape, dtype=x.dtype, layout=x.layout, device=x.device) for x in  example_inputs])
    input_names = tuple([f"input_{i}" for i in range(len(real_inputs))])
    import os
    os.makedirs("tmp", exist_ok=True)
    model_name = sys_config.get_config('model_name')
    model_path = f"tmp/{model_name}_onnx_graph_{num_graph}.onnx"
    from fx2onnx import to_onnx
    import onnx
    onnx_graph = to_onnx(gm, *real_inputs)
    onnx.save(onnx_graph, model_path)
    
    # run with onnx
    # import onnxruntime as ort
    # ort_session = ort.InferenceSession(model_path)
    # input_names = [inp.name for inp in ort_session.get_inputs()]

    # def run_with_onnx(*args):
    #     print("-- run with onnx")
    #     import numpy as np
    #     inputs = [x.cpu().numpy() for x in args]
    #     input_names = [inp.name for inp in ort_session.get_inputs()]
    #     ort_inputs = dict(zip(input_names, inputs))
    #     # print("ort_inputs", ort_inputs)
    #     outputs = ort_session.run(None, ort_inputs)
    #     outputs = [torch.tensor(x).cuda() for x in outputs]
    #     expect_outputs = gm.forward(*args)
    #     for i in range(len(expect_outputs)):
    #         # print("evaluate", i, flush=True)
    #         assert_equal(expect_outputs[i], outputs[i])    
    #     return outputs
    # return run_with_onnx

    import onnx
    import onnxruntime as ort
    def load_ort_session(model_path):
        onnx_model = onnx.load(model_path)
        print(onnx.helper.printable_graph(onnx_model.graph))
        # print(onnx_model.graph.value_info)
        onnx.checker.check_model(onnx_model)
        print(f"{model_path}: check passed!")
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        session = ort.InferenceSession(model_path)

        inputs_name = [item.name for item in onnx_model.graph.input]
        outputs_name = [item.name for item in onnx_model.graph.output]
        return session, inputs_name, outputs_name
    
    # NNFUSION_ROOT = os.path.expanduser("~/frontend/nnfusion")
    # os.environ["PATH"] = os.path.abspath(NNFUSION_ROOT) + ":" + os.environ["PATH"]
    # sys.path.insert(1, os.path.abspath(NNFUSION_ROOT + "/src/python"))
    from nnfusion.session import codegen, modify_nnfusion_rt, build
    from nnfusion.executor import Executor
    from nnfusion.data_format import cast_pytorch_tensor
    def build_nnfusion(onnx_model_path, codegen_flags, workdir, rt_dir):
        flags_str = "-f onnx "
        flags_str += " ".join([
            "-f{}={}".format(k, v) for k, v in codegen_flags.items()
        ])
        # print("work dir:", workdir,)
        os.system(f"rm -r {workdir}")
        os.system(f"mkdir -p {workdir}")
        os.system(f"cp -r tmp/bin {workdir}")
        codegen(onnx_model_path, flags_str, workdir)
        # os.system(f"cat {workdir}/codegen.log ")
        modify_nnfusion_rt(rt_dir)
        build(rt_dir)
    def load_executor(model_path: str):
        assert(model_path.endswith('.onnx'))
        workdir = os.path.abspath(model_path[:-5])
        codegen_flags = {
            "autodiff": False,  # add backward graph
            "training_mode": False,  # move weight external
            "extern_result_memory": True, # move result external
            "codegen_unexist_kernel": True, # generate kernel for unexist op
            "product_name": "A100",
            "default_device": "CUDA",
            "kernel_cache_path": f'/tmp/{os.environ.get("USER")}/nnfusion/kernel_cache.db',
            'biasadd_fix': True,
            'check_result': True,
            'conv_cnhw': True,
            'max_grid_dim': 256,
            'cf_level': 2,
            'branch_fine_grained': False,
            'branch_split': False,
            'log_kerneldb_request': False
        }
        rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
        build_nnfusion(model_path, codegen_flags, workdir, rt_dir)
        executor = Executor(rt_dir)
        return executor

    executor = load_executor(model_path)

    def fn(*args):
        inputs = [cast_pytorch_tensor(item) for item in args]
        input_signatures = [x.pointer_type for x in inputs]
        input_pointers = [x.pointer for x in inputs]
        output_tensors = executor.alloc_output_buffer()
        output_casted = [cast_pytorch_tensor(x) for x in output_tensors]
        output_signatures = [x.pointer_type for x in output_casted]
        output_pointers = [x.pointer for x in output_casted]
        signatures = input_signatures + output_signatures
        pointers = input_pointers + output_pointers
        executor.feed_pointers(signatures, pointers)
        return output_tensors
    return fn


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
    print("compile_mode:", compile_mode)
    timer.report()
    # nsys will kill proc after profile_stop
    profile_stop()


def perf_test_with_profile(f, graph_timer, compile_mode, repeat, args, kwargs):
    torch.cuda.synchronize()
    start_time = time.time()
    o = f(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    print("first run", end_time - start_time, "s")

    for idx in range(repeat - 1):
        torch.cuda.synchronize()
        o = f(*args, **kwargs)
        torch.cuda.synchronize()
    graph_timer.clear()
    timer = Timer("ms")
    for idx in range(repeat):
        torch.cuda.synchronize()
        timer.start()
        o = f(*args, **kwargs)
        torch.cuda.synchronize()
        timer.log()
    print("compile_mode:", compile_mode)
    timer.report(text = 'e2e')
    graph_timer.report(text = 'graph profile')


def perf_test_run_cf(f, compiled, compile_mode, repeat, args_all, kwargs_all):
    for idx in range(repeat):
        o1 = f(*args_all[idx], **kwargs_all[idx])
        torch.cuda.synchronize()
        o2 = compiled(*args_all[idx], **kwargs_all[idx])
        torch.cuda.synchronize()
        assert_equal(o1, o2)

    profile_start()
    timer = Timer('ms')
    for idx in range(repeat):
        # print("run:", idx)
        torch.cuda.synchronize()
        timer.start()
        o = compiled(*args_all[idx], **kwargs_all[idx])
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
    elif compile_mode == "script":
        compiled = f
    elif compile_mode == "dynamo":
        torch._dynamo.reset()
        compiled = torch.compile(f)
    elif compile_mode == "dynamo-tensorrt":
        import torch_tensorrt
        torch._dynamo.reset()
        compiled = torch_tensorrt.dynamo.compile(f, args)
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
    elif compile_mode == "dynamo-graph":
        torch._dynamo.reset()
        # explain(f, *args, **kwargs)
        # torch._dynamo.reset()
        compiled = torch.compile(f, backend=custom_backend)
    elif compile_mode == "dynamo-onnx":
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=onnx_backend)
    elif compile_mode == "dynamo-nnf":
        torch._dynamo.reset()
        compiled = torch.compile(f, backend=nnf_backend)
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
    elif compile_mode == "sys-profile":
        graph_timer = Timer("ms")
        compiler = get_inductor_with_profile(graph_timer)
        sys_config.set_config('backend', compiler)
        sys_config.set_config('debug', False)
        # print("compiler:", compiler, flush=True)
        # print("is_debug", sys_config.get_config('debug'))
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
    elif compile_mode == "sys-nnf":
        sys_config.set_config('debug', False)
        def compile_fn(gm, example_inputs):
            model_name = sys_config.get_config('model_name')
            from fx2onnx import compile_with_nnf  # type: ignore[import]
            from frontend.fx_graph import generate_real_tensors
            real_inputs = generate_real_tensors(example_inputs)
            return compile_with_nnf(model_name, gm, real_inputs)
        sys_config.set_config('backend', compile_fn)
        compiled = sys_compile(f)
    elif compile_mode == "sys-torchscript":
        sys_config.set_config('debug', False)
        sys_config.set_config('backend', 'script')
        compiled = sys_compile(f)
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
    if compile_mode == "dynamo-graph" or compile_mode == "dynamo-onnx":
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
    elif compile_mode == "sys-profile":
        perf_test_with_profile(compiled, graph_timer, compile_mode, num_repeat, args, kwargs)
    else:
        perf_test_run(compiled, compile_mode, num_repeat, args, kwargs)

    if compile_mode == "dynamo-graph":
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


def script_with_log(*args, **kwargs):
    print("run torch.jit.script")
    return torch.jit.script(*args, **kwargs)