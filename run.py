import logging
logging.basicConfig(level=logging.WARNING)
from frontend.no_preload import NO_LD_PRELOAD_CTX
with NO_LD_PRELOAD_CTX():
    from utils import perf_test
    import argparse
    import importlib
    # import models
    import random
    import torch
    import numpy as np
    import torch_xla.core.xla_model as xm
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", type=str, default="sys")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--dyn_cf", action="store_true")
    parser.add_argument("--dyn_bs", action="store_true")
    parser.add_argument("--dyn_len", action="store_true")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--no_check", dest="check", action="store_false")

    args = parser.parse_args()
    print(args)

    random_seed = 23333
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    def main():
        module = importlib.import_module("."+args.model, package="models")
        if args.compile == "script":
            assert hasattr(module, 'get_scripted_model')
            model = module.get_scripted_model()
        elif hasattr(module, 'get_model'):
            model = module.get_model()
        elif hasattr(module, 'get_model_with_bs'):
            model = module.get_model_with_bs(args.bs)
        else:
            raise ValueError("lack of get_model in {}".format(args.model))
        model.eval()
        if args.dyn_cf + args.dyn_bs + args.dyn_len == 0:
            input_args, input_kwargs = module.get_input(batch_size=args.bs)
            expected_output = model(*input_args, **input_kwargs)
        else:
            expected_output = None
        assert args.dyn_cf + args.dyn_bs + args.dyn_len <= 1
        import frontend
        frontend.config.set_config('model_name', f"{args.model}_bs{args.bs}")
        if args.compile in ("xla", "dynamo-xla", "sys-xla"):
            model = model.to('cpu').to(xm.xla_device())
        if args.dyn_cf:
            assert hasattr(module, 'get_dynamic_inputs')
            input_args, input_kwargs = module.get_dynamic_inputs(args.bs, 2 * args.repeat)
            if args.model == 'blockdrop':
                frontend.dynamic.add_branch_rewrite_pc(frontend.c_api.get_next_frame_id(), 51)
            if args.model == 'lstm':
                for_iter_pc = 32
                frontend.dynamic.mark_dynamic_pc(frontend.c_api.get_next_frame_id(), for_iter_pc,
                                frontend.dynamic.DynamicControlFlow(for_iter_pc, "FOR_ITER"))
            perf_test(model, args.compile, input_args, input_kwargs, None, module.get_input, args.repeat, 'cf', args.check)
        elif args.dyn_bs:
            perf_test(model, args.compile, None, None, None, module.get_input, args.repeat, 'bs', args.check)
        elif args.dyn_len:
            perf_test(model, args.compile, None, None, None, module.get_input, args.repeat, 'len', args.check)
        else:
            perf_test(model, args.compile, input_args, input_kwargs, expected_output, module.get_input, args.repeat, None, args.check)

    if __name__ == "__main__":
        with torch.no_grad():
            main()
    
