import os
import re
import numpy as np
import pandas as pd

# static

log_dir = 'logs/231105-161002'
pattern = r'100 iters, min = [\d.]+ s, max = [\d.]+ s, avg = ([\d.]+) s'

model_list = ('align', 'bart', 'deberta', 'densenet', 'monodepth', 'quantized', 'tridentnet')
syss = ('eager', 'dynamo',  'sys',  'xla', 'dynamo-xla', 'sys-xla')
for bs in (1, 16):
    results = np.full((len(model_list), len(syss)), np.nan)
    for i, models in enumerate(model_list):
        for j, sys in enumerate(syss):
            filename = f'{models}.{bs}.{sys}.log'
            if not os.path.exists(os.path.join(log_dir, filename)):
                print(filename, "not exists")
                continue
            with open(os.path.join(log_dir, filename), 'r') as file:
                content = file.read()
            match = re.search(pattern, content)
            if match:
                avg_time = float(match.group(1))
                results[i, j] = avg_time
            else:
                print(filename, "run fail")
    results_df = pd.DataFrame(results, index=model_list, columns=syss)
    results_df.to_csv(f'{log_dir}/results.{bs}.csv')

# dynamic bs

log_dir = 'logs/dyn-bs-231126-201524'
pattern = r'90 iters, min = [\d.]+ s, max = [\d.]+ s, avg = ([\d.]+) s'

model_list = ('bert', 'bert-seqlen', 'resnet', 'deberta', 'deberta-seqlen', 'densenet')
syss = ('eager', 'dynamo-dynamic',  'sys-dynamic')

results = np.full((len(model_list), len(syss)), np.nan)
for i, models in enumerate(model_list):
    for j, sys in enumerate(syss):
        filename = f'{models}..{sys}.log'
        if not os.path.exists(os.path.join(log_dir, filename)):
            print(filename, "not exists")
            continue
        with open(os.path.join(log_dir, filename), 'r') as file:
            content = file.read()
        match = re.search(pattern, content)
        if match:
            avg_time = float(match.group(1))
            results[i, j] = avg_time
        else:
            print(filename, "run fail")
results_df = pd.DataFrame(results, index=model_list, columns=syss)
results_df.to_csv(f'{log_dir}/results.{bs}.csv')
