# SDOclust

## Overview
This project aims to explore strategies for parallelizing SDOclust.  
The main focus is on parallelizing the label extension phase, which is often the main computational bottleneck when clustering large datasets.  

The implemented approach follows a split-based strategy:  
1. Run SDOclust on a subset of the data (split-A) to obtain observers and their cluster labels.  
2. Compress the observer set to reduce computational cost.  
3. Extend cluster labels from observers to the full dataset.  
4. Execute the label extension step using different execution models:
    - Sequential  
    - Joblib (multi-threaded)  
    - Dask (distributed / multi-process)  

## Methology
1. Observer Extraction  
  After fitting an SDOclust model on split-A, the observers and their corresponding labels are extracted from the fitted model object.
  This step is implemented in a utility function that inspects the internal attributes of the model.

2. Label Extension Methods  
  Two label extension strategies are implemented:   
    (a) Radius Voting (DBSCAN-like): Each data point is assigned a label based on majority voting among observers within a fixed radius.  
    (b) 1-NN + Radius Cutoff: Each data point is assigned the label of its nearest observer if the distance is below a threshold.
    
    The method (b) is faster but less likely to DBSCAN behavior.

3. Parallel Execution Backends  
  Each label extension strategy is executed using three different backends:  
    - Sequential: baseline implementation  
    - Joblib (threads backend): multi thread execution on a single machine
    - Dask (distributed): parallel execution using a Dask distributed client

This allows direct comparison of performance and scalability across execution models.

## Files
```
run_parallel_sdoclust.py
```
Main script that runs SDOclust on split-A and benchmarks parallel label extension methods.
```
dbscan_extension.py
```
Contains label extension logic, observer compression functions, and parallel implementations.

## Evaluation

Clustering quality and performance are evaluated using:

- Adjusted Rand Index (ARI)  
- Adjusted Mutual Information (AMI)  
- Runtime  
- Fraction of points labeled as noise

Results are reported for each execution backend to assess the trade-offs between speed and clustering accuracy.

## How to Run
To run full pipeline: 
```
python3 run_parallel_sdoclust.py
```

To run benchmark the label extension methods only:
```
python3 dbscan_extension.py
```

## Results
```
SDOclust fit done in 0.231s
Observers: O (256, 10), labels: ol (256,)
Radius voting (seq)          Time=0.120s  ARI=0.978  AMI=0.965  noise=0.017
Radius voting (joblib)       Time=0.121s  ARI=0.978  AMI=0.965  noise=0.017
Radius voting (dask)         Time=0.077s  ARI=0.978  AMI=0.965  noise=0.017
1-NN + eps (FAST)            Time=0.043s  ARI=0.978  AMI=0.965  noise=0.017
1-NN + eps (joblib)          Time=0.049s  ARI=0.978  AMI=0.965  noise=0.017
1-NN + eps (dask)            Time=0.080s  ARI=0.978  AMI=0.965  noise=0.017
```
#### Compare with baseline
```
SDOclust                 Time=1.089s  ARI=1.000  AMI=1.000
SDOclust (chunks)        Time=0.934s  ARI=1.000  AMI=1.000
```
