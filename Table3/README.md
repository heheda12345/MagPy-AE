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
  -untested models:   579
    --no tests:       490
    --all crashed:    15
    --no profiling:   74

  -remaining models:        1421
  -dynamic overall models:  230
     --eager dynamic:       110
     --graph dynamic:       120
  -static models:     1191

mode         total      Failed cases  Fail rate
sys          1191       79         6.6%    
dynamo       1191       272        22.8%    
torchscript  1191       769        64.6%
```
