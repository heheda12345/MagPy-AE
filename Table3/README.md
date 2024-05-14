## Evaluating and Profiling ParityBench result

### 1. Preparing

To evaluate and profile the results shown in Table 3, please download the following ParityBench source code and do the experiments under the directory:

```
git clone git@github.com:heheda12345/pytorch-jit-paritybench.git
cd pytorch-jit-paritybench
```

### 2. Evaluating 

**For all models in this ParityBench**


We currenly support three ```dynamo```, ```sys```, ```torchscript``` compilation modes in this benchmark


**To evaluate dynamo:**
```
./evaluate-all.sh test-list.txt dynamo
```

**To evaluate Torchsript:**
```
./evaluate-all.sh test-list.txt torchscript
```

**To evaluate DeepVisor:**
```
./evaluate-all.sh test-list.txt sys
```

### 3. Profiling

After evaluating all models for each compilation mode, the verbose results will be generated in ```logs/``` within ```TIME_TAGGED``` directory for each mode. 

To simplify the profiling, it is better to run profiling every time after you run and finish a compilation mode evaluation.

**To profile results:**
```
./profile.sh logs/your_generated_directory/ compilation_mode
```

for example, the following command can be used to profile ```DeepVisor``` results, and the ```TIME_TAGGED``` directory should be the time when you starting to evaluating the benchmark:
```
./profile.sh logs/240509-200914/ sys
```
