from utils import perf_test
import argparse
import importlib
# import models
import random
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--compile", type=str, default="sys")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--dynamic", action="store_true")
parser.add_argument("--repeat", type=int, default=100)

args = parser.parse_args()
print(args)

random_seed = 23333
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

def main():
    module = importlib.import_module("."+args.model, package="models")
    if hasattr(module, 'get_model'):
        model = module.get_model()
    elif hasattr(module, 'get_model_with_bs'):
        model = module.get_model_with_bs(args.bs)
    else:
        raise ValueError("lack of get_model in {}".format(args.model))
    model = model.eval()
    if args.dynamic and hasattr(module, 'get_dynamic_inputs'):
        input_args, input_kwargs = module.get_dynamic_inputs(args.bs, 2 * args.repeat)
        perf_test(model, args.compile, input_args, input_kwargs, args.repeat, True)
    else:
        input_args, input_kwargs = module.get_input(batch_size=args.bs)
        perf_test(model, args.compile, input_args, input_kwargs, args.repeat, False)

if __name__ == "__main__":
    with torch.no_grad():
        main()
    