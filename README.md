# SDOclust

## Overview
This project aims to implement parallelizable architectures for SDOclust.
The main computational bottleneck of SDOclust is the label extension phase, which assigns cluster labels to all data points based on the observer model.
This project focuses on parallelizing this phase using different execution backends.

## Methology
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
Baseline SDOclust Time=1.212s  ARI=1.000  AMI=1.000

======================================================================
[FIXED n_splits] [backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  25000   0.501   0.134   0.635    265  1.000  1.000
    seq      4  12500   0.266   0.112   0.378    261  1.000  1.000
    seq      8   6250   0.152   0.169   0.321    252  1.000  1.000
    seq     16   3125   0.079   0.172   0.251    240  1.000  1.000

======================================================================
[FIXED n_splits] [backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  25000   0.464   0.141   0.605    265  1.000  1.000
 joblib      4  12500   0.257   0.076   0.333    261  1.000  1.000
 joblib      8   6250   0.165   0.110   0.274    252  1.000  1.000
 joblib     16   3125   0.076   0.068   0.144    240  1.000  1.000

======================================================================
[FIXED n_splits] [backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  25000   0.440   0.322   0.761    265  1.000  1.000
   dask      4  12500   0.265   0.063   0.327    261  1.000  1.000
   dask      8   6250   0.147   0.075   0.222    252  1.000  1.000
   dask     16   3125   0.064   0.082   0.146    240  1.000  1.000

======================================================================
[splitA-frac] [backend=seq] knn=10 chunksize=2000 rest_splits=8
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq   0.02   1000   0.040   0.158   0.197    190  1.000  1.000
    seq   0.05   2500   0.062   0.140   0.202    231  1.000  1.000
    seq   0.10   5000   0.115   0.188   0.303    252  1.000  1.000
    seq   0.20  10000   0.215   0.145   0.360    257  1.000  1.000
    seq   0.40  20000   0.375   0.113   0.488    263  1.000  1.000
    seq   1.00  50000   0.995   0.136   1.130    267  1.000  1.000

======================================================================
[splitA-frac] [backend=joblib] knn=10 chunksize=2000 rest_splits=8
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib   0.02   1000   0.039   0.102   0.141    194  1.000  1.000
 joblib   0.05   2500   0.072   0.084   0.156    234  1.000  1.000
 joblib   0.10   5000   0.104   0.082   0.185    253  1.000  1.000
 joblib   0.20  10000   0.187   0.095   0.282    258  1.000  1.000
 joblib   0.40  20000   0.386   0.078   0.464    262  1.000  1.000
 joblib   1.00  50000   0.916   0.115   1.030    266  1.000  1.000

======================================================================
[splitA-frac] [backend=dask] knn=10 chunksize=2000 rest_splits=8
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask   0.02   1000   0.052   0.091   0.143    203  1.000  1.000
   dask   0.05   2500   0.054   0.103   0.157    234  1.000  1.000
   dask   0.10   5000   0.112   0.068   0.180    251  1.000  1.000
   dask   0.20  10000   0.196   0.094   0.290    256  1.000  1.000
   dask   0.40  20000   0.390   0.060   0.450    263  1.000  1.000
   dask   1.00  50000   0.919   0.130   1.049    267  1.000  1.000
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
