# SDOclust

## Overview
This project aims to implement parallelizable architectures for SDOclust.
The main computational bottleneck of SDOclust is the label extension phase, which assigns cluster labels to all data points based on the observer model.
This project focuses on parallelizing this phase using different execution backends.

## Methodology
1. Data Splitting  
The dataset is divided into multiple splits of similar size.

    Two experimental settings are implemented:
    - Fixed n-Splits: the dataset is divided into n equal splits, one used as split-A.  
    - Split-A Fraction Sweep: split-A size is varied as a fraction of the dataset.  
  
3. Observer Model (Split-A)   
SDOclust is run only on one split (split-A).  
From the fitted model, observers and observer cluster labels are extracted.  
These observers are then shared with all other splits.

4. Parallel Execution Backends  
The label extension phase is executed using different execution models:   
    - per-batch processing: baseline implementation  
    - joblib: multi-thread execution on a single machine  
    - dask: task-based parallel execution
          - dask-local: local based (single machine)  
          - dask-dist: distributed based (cluster ready)

    Each backend executes label extension independently on each split, and results are merged back into the original data order.

## Files
```
sdoclust_parallel.py
```
Main script implementing the split-based SDOclust architecture and running all experiments.

Includes:  
- fixed n-splits experiments  
- split-A fraction experiments  
- sequential / joblib / dask comparisons  
- baseline SDOclust comparison  

## Evaluation
Clustering quality and performance are evaluated using:  
- Adjusted Rand Index (ARI)  
- Adjusted Mutual Information (AMI)  
- Runtime  
    - SDOclust fitting time (fit)  
    - Label extension time (ext)  
    - Total runtime (total)    
- Number of observers (n_obs)

Results are reported in table:
- scalability with respect split size  
- performance differences between execution backends  
- trade offs between computation time and clustering quality  

## How to Run
```
python3 ./sdoclust_parallel.py
```

## Results
```
Baseline SDOclust Time=1.091s  ARI=1.000  AMI=1.000

======================================================================
[FIXED n_splits] [backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  25000   0.521   0.133   0.654    265  1.000  1.000
    seq      4  12500   0.259   0.120   0.379    261  1.000  1.000
    seq      8   6250   0.185   0.158   0.343    252  1.000  1.000
    seq     16   3125   0.083   0.188   0.271    240  1.000  1.000

======================================================================
[FIXED n_splits] [backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  25000   0.497   0.140   0.637    265  1.000  1.000
 joblib      4  12500   0.267   0.069   0.336    261  1.000  1.000
 joblib      8   6250   0.157   0.089   0.245    252  1.000  1.000
 joblib     16   3125   0.077   0.090   0.167    240  1.000  1.000

======================================================================
[FIXED n_splits] [backend=dask-local] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
dask-local      2  25000   0.457   0.166   0.623    265  1.000  1.000
dask-local      4  12500   0.251   0.115   0.366    261  1.000  1.000
dask-local      8   6250   0.137   0.158   0.295    252  1.000  1.000
dask-local     16   3125   0.074   0.180   0.254    240  1.000  1.000

======================================================================
[FIXED n_splits] [backend=dask-dist] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
dask-dist      2  25000   0.478   0.160   0.638    265  1.000  1.000
dask-dist      4  12500   0.265   0.106   0.372    261  1.000  1.000
dask-dist      8   6250   0.139   0.179   0.318    252  1.000  1.000
dask-dist     16   3125   0.096   0.251   0.347    240  1.000  1.000

======================================================================
[splitA-frac] [backend=seq] knn=10 chunksize=2000 rest_splits=8
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq   0.02   1000   0.031   0.173   0.204    190  1.000  1.000
    seq   0.05   2500   0.069   0.183   0.252    231  1.000  1.000
    seq   0.10   5000   0.119   0.202   0.321    252  1.000  1.000
    seq   0.20  10000   0.193   0.141   0.334    257  1.000  1.000
    seq   0.40  20000   0.409   0.115   0.524    263  1.000  1.000
    seq   1.00  50000   0.979   0.141   1.120    267  1.000  1.000

======================================================================
[splitA-frac] [backend=joblib] knn=10 chunksize=2000 rest_splits=8
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib   0.02   1000   0.033   0.088   0.121    194  1.000  1.000
 joblib   0.05   2500   0.197   0.113   0.310    234  1.000  1.000
 joblib   0.10   5000   0.103   0.069   0.172    253  1.000  1.000
 joblib   0.20  10000   0.246   0.058   0.304    258  1.000  1.000
 joblib   0.40  20000   0.434   0.079   0.513    262  1.000  1.000
 joblib   1.00  50000   0.961   0.116   1.077    266  1.000  1.000

======================================================================
[splitA-frac] [backend=dask-local] knn=10 chunksize=2000 rest_splits=8
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
dask-local   0.02   1000   0.032   0.157   0.189    203  1.000  1.000
dask-local   0.05   2500   0.071   0.137   0.208    234  1.000  1.000
dask-local   0.10   5000   0.101   0.138   0.239    251  1.000  1.000
dask-local   0.20  10000   0.216   0.130   0.346    256  1.000  1.000
dask-local   0.40  20000   0.357   0.113   0.470    263  1.000  1.000
dask-local   1.00  50000   0.933   0.131   1.064    267  1.000  1.000

======================================================================
[splitA-frac] [backend=dask-dist] knn=10 chunksize=2000 rest_splits=8
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
dask-dist   0.02   1000   0.039   0.180   0.218    194  1.000  1.000
dask-dist   0.05   2500   0.059   0.227   0.287    239  1.000  1.000
dask-dist   0.10   5000   0.111   0.172   0.283    251  1.000  1.000
dask-dist   0.20  10000   0.241   0.129   0.371    257  1.000  1.000
dask-dist   0.40  20000   0.423   0.153   0.575    263  1.000  1.000
dask-dist   1.00  50000   0.931   0.112   1.042    267  1.000  1.000

```
### Table Notation
- **splitA**: how split-A is defined   
    - number of splits (n) in the fixed n-splits setting  
    - fraction (f) in the split-A fraction sweep      
- **|A|**: the number of samples used in split-A to fit the SDOclust model   
- **fit**: time required to fit SDOclust on split-A   
- **ext**: time spent on label extension  
- **total**: sum of fitting and extension times  
- **n_obs**: Number of observers  
- **ARI**: Adjusted Rand Index   
- **AMI**: Adjusted Mutual Information
