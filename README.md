# Cloud compute calculator

query.csv
```
cpu,mem,gpu,gpu_mem,total_hours
4,32,0,0,100
32,128,0,0,1000
```

```
python calculator.py query.csv
```

```
INFO:__main__:Your query:
INFO:__main__:   cpu  mem  gpu  gpu_mem  total_hours
0    4   32    0        0          100
1   32  128    0        0         1000
INFO:__main__:Detailed costs:                                                       total_hours  total
offer                    cpu mem   gpu gpu_mem price
linux-d32av4-lowpriority 32  128.0 0   0       0.358         1000  358.0
linux-e4v4-lowpriority   4   32.0  0   0       0.056          100    5.6
INFO:__main__:Total cost (USD): 363.6
```
