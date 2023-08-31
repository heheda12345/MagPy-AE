from utils import perf_test
import argparse
import importlib
# import models
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument("--compile", type=str, default="sys")
argparser.add_argument("--model", type=str, required=True)
argparser.add_argument("--bs", type=int, default=1)

args = argparser.parse_args()

def main():
    module = importlib.import_module("."+args.model, package="models")
    if hasattr(module, 'get_model'):
        model = module.get_model()
    elif hasattr(module, 'get_model_with_bs'):
        model = module.get_model_with_bs(args.bs)
    else:
        raise ValueError("lack of get_model in {}".format(args.model))
    model = model.eval()
    input_args, input_kwargs = module.get_input(batch_size=args.bs)
    with torch.no_grad():
        perf_test(model, args.compile, input_args, input_kwargs)

if __name__ == "__main__":
    main()
    