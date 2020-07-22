# Cloud compute calculator

query.csv
```
cpu,mem,gpu,gpu_mem,total_hours
4,32,0,0,100
32,128,0,0,1000
```

```
python calculator.py query.csv --region us-west us-east --pricing-scheme perhour perhourspot
```

```
INFO:__main__:Regions: ['us-west', 'us-east']
INFO:__main__:Tiers: ['standard']
INFO:__main__:Pricing schemes: ['perhour', 'perhourspot']
INFO:__main__:Currency: usd
INFO:__main__:Your query:
   cpu  mem  gpu  gpu_mem  total_hours
0    4   32    0        0          100
1   32  128    0        0         1000
INFO:__main__:Detailed costs (sorted from lowest to highest total):
    region                  offer pricing_scheme  cpu    mem  gpu  gpu_mem     price  total_hours     total
0  us-east   linux-a4mv2-standard    perhourspot    4   32.0    0        0  0.033844          100    3.3844
1  us-east  linux-d32av4-standard    perhourspot   32  128.0    0        0  0.274330         1000  274.3300
INFO:__main__:Total cost (USD): 277.71
```
