from utils import perf_test
import argparse
import importlib
import models

argparser = argparse.ArgumentParser()
argparser.add_argument("--compile", type=str, default="sys")
argparser.add_argument("--model", type=str, required=True)
argparser.add_argument("--bs", type=int, default=1)

args = argparser.parse_args()

def main():
    module = importlib.import_module("."+args.model, package="models")
    model = module.get_model()
    input_args, input_kwargs = module.get_input(batch_size=args.bs)
    perf_test(model, args.compile, input_args, input_kwargs)

if __name__ == "__main__":
    main()
    