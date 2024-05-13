LOG_DIR = '../logs/graphcount'

model_names = ['align', 'bert', 'deberta', 'densenet', 'monodepth' ,'quantized', 'resnet', 'tridentnet']
for model in model_names:
    dynamo_cnt = 0
    torchscript_cnt = 0
    with open(f"{LOG_DIR}/{model}.dynamo-graph.log") as f:
        for s in f.readlines():
            if s.startswith("num_graph:"):
                dynamo_cnt = int(s.strip().split()[-1])
    with open(f"{LOG_DIR}/{model}.script.log") as f:
        for s in f.readlines():
            if s.strip() == "run torch.jit.script":
                torchscript_cnt += 1
    print(f"{model=}, {dynamo_cnt=}, {torchscript_cnt=}")

