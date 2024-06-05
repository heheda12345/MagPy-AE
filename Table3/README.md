## Evaluating and Analyzing ParityBench result

### 1. Preparing

To evaluate and get the results shown in Table 3, please download ParityBench source code and do the experiments under the directory:
**For AE Reviewers**: This repo is saved at `~/pytorch-jit-paritybench`

```
git clone git@github.com:heheda12345/pytorch-jit-paritybench.git
cd pytorch-jit-paritybench
```

### 2. Evaluating 
The following command evaluates dynamo, torchscript and MagPy with all models in ParityBench. Note that this command will takes a long time (Ten hours if using one GPU)

```
./run.sh
```


### 3. Analyzing

After evaluating all models for each compilation mode, the verbose results will be generated in ```logs/``` within ```compilation mode``` directory for each mode. 


**To dump overall results:**
```
./analyze.sh
```

you will get below results:
```
analyzing...
overall models 2000
  -untested models: 581
    --no tests:      507
    --no profiling:  74

  -remaining models:  1419
  -dynamic models:    128
  -static models:     1291

mode         total      Failed cases  Fail rate
sys          1291       89         6.9%    
dynamo       1291       838        39.3%    
torchscript  1291       507        64.9%
```
