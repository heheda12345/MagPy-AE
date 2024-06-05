import torch
from .magic_rewrite import magic_rewrite
    
def compile_with_nnf(model_name: str, gm: torch.fx.GraphModule,
                    real_inputs: list[torch.Tensor]):
    from fx2onnx import to_onnx
    import onnx
    import os, sys
    
    gm = magic_rewrite(gm)
    onnx_graph = to_onnx(gm, *real_inputs)
    # run with onnx
    # import onnxruntime as ort
    # ort_session = ort.InferenceSession(onnx_graph.SerializeToString())
    # input_names = [inp.name for inp in ort_session.get_inputs()]
    # def run_with_onnx(*args):
    #     print("run with onnx")
    #     import numpy as np
    #     inputs = [x.cpu().numpy() for x in args]
    #     input_names = [inp.name for inp in ort_session.get_inputs()]
    #     ort_inputs = dict(zip(input_names, inputs))
    #     # print("ort_inputs", ort_inputs)
    #     outputs = ort_session.run(None, ort_inputs)
    #     return [torch.tensor(x) for x in outputs]
    # return run_with_onnx
    # run with onnx end
    NNFUSION_ROOT = os.path.expanduser('~/frontend/nnfusion')
    os.environ["PATH"] = os.path.abspath(NNFUSION_ROOT) + ":" + os.environ["PATH"]
    sys.path.insert(1, os.path.abspath(NNFUSION_ROOT + "/src/python"))

    from nnfusion.session import codegen, modify_nnfusion_rt, build
    from nnfusion.executor import Executor
    from nnfusion.data_format import cast_pytorch_tensor

    def build_nnfusion(onnx_model_path, codegen_flags, workdir, rt_dir):
        os.system(f"rm -r {workdir}")
        os.system(f"mkdir -p {workdir}")
        codegen(onnx_model_path, codegen_flags, workdir)
        # os.system(f"cat {workdir}/codegen.log ")
        modify_nnfusion_rt(rt_dir)
        build(rt_dir)

    def load_executor(model_path: str, codegen_flags: str):
        assert(model_path.endswith('.onnx'))
        workdir = os.path.abspath(model_path[:-5])
        rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
        build_nnfusion(model_path, codegen_flags, workdir, rt_dir)
        executor = Executor(rt_dir)
        return executor
    # gm = magic_rewrite(model_name, gm)

    model_path = f'tmp/{model_name}.onnx'
    onnx.save(onnx_graph, model_path)
    kernel_cache_path = f'/tmp/{os.environ.get("USER")}/nnfusion/kernel_cache.db'
    model_compile_config = {
        'lstm_bs1': f'-f onnx -fcodegen_unexist_kernel=true -fproduct_name=A100 -fbiasadd_fix=true -fcheck_result=true -fextern_result_memory=true -fconv_cnhw=false -fdefault_device=CUDA -fkernel_cache_path={kernel_cache_path} -fcf_level=2',
        'lstm_bs16': f'-f onnx -fcodegen_unexist_kernel=true -fproduct_name=A100 -fbiasadd_fix=true -fcheck_result=true -fextern_result_memory=true -fconv_cnhw=false -fdefault_device=CUDA -fkernel_cache_path={kernel_cache_path} -fcf_level=2',
        'blockdrop_bs1': f'-f onnx -fcodegen_unexist_kernel=true -fproduct_name=A100 -fbiasadd_fix=true -fcheck_result=true -fextern_result_memory=true -fconv_cnhw=true -fdefault_device=CUDA -fkernel_cache_path={kernel_cache_path} -fcf_level=1 -fbranch_split=false -fbranch_fine_grained=false -fmax_grid_dim=160 -fmax_block_dim=256',
        'blockdrop_bs16': f'-f onnx -fcodegen_unexist_kernel=true -fproduct_name=A100 -fbiasadd_fix=true -fcheck_result=true -fextern_result_memory=true -fconv_cnhw=true -fdefault_device=CUDA -fkernel_cache_path={kernel_cache_path} -fcf_level=1 -fbranch_split=true -fbranch_fine_grained=true -fmax_grid_dim=128 -fmax_block_dim=512'
    }
    if model_name not in model_compile_config:
        raise NotImplementedError
    executor = load_executor(model_path, model_compile_config[model_name])
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