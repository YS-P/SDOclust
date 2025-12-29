# SDOclust

## Overview
This project aims to implement parallelizable architectures for SDOclust.
One of the main bottlenecks of SDOclust is the label extension phase, which assigns cluster labels to all data points based on the observer model.
This project focuses on parallelizing this phase using different execution backends.

## Methodology
1. Data Splitting  
The dataset is divided into multiple splits of similar size.
    - Dataset is divided into n equal splits, one used as split-A.
  
2. Observer Model (Split-A)   
- SDOclust is run only on one split (split-A).  
- From the fitted model, observers and observer cluster labels are extracted.  
- These observers are then shared with all other splits.  
- Label extension is performed independently on each remaining split.  
- Predicted labels are merged back to recover the full clustering.

3. Parallel Execution Backends  
The label extension phase is executed using different execution models:   
    - per-batch processing: baseline implementation  
    - joblib: multi-thread execution on a single machine  
    - dask: task-based parallel execution  

    Each backend executes label extension independently on each split, and results are merged back into the original data order.

## Evaluation
Clustering quality and performance are evaluated using:  
- Adjusted Rand Index (ARI)  
- Adjusted Mutual Information (AMI)  
- Runtime  
    - SDOclust fitting time (fit)  
    - Label extension time (ext)  
    - Total runtime (total)    
- Number of observers (n_obs)

For noisy_blobs, noise points are labeled -1 and excluded from ARI/AMI computation (metrics are computed on y_true != -1).

Results are reported in table:
- Scalability with respect split size  
- Performance differences between execution backends  
- Trade offs between computation time and clustering quality


## Experimental Setup  
All experiments are conducted on synthetic Gaussian blob datasets generated using `sklearn.datasets.make_blobs`.

Unless otherwise stated, the following default parameters are used:  
- Number of clusters: 5  
- Feature dimensions: 10 and 50  
- Dataset sizes: up to 200,000 samples  
- kNN neighbors for label extension: k = 10  
- Standard deviation of clusters: 1.0 and 2.0  

Each experiment is repeated using fixed random seeds to ensure reproducibility.

### Datasets
Experiments are run on two synthetic dataset variants:  

- blobs: clean Gaussian blobs generated with `make_blobs`.  
- noisy_blobs: the same blobs dataset + uniform noise points.  
  Noise points are sampled uniformly within the feature min/max range of the blob data.  
  These noise points are assigned the ground truth label '-1'.

## Files
```
sdoclust_parallel.py
```
Main script implementing the split-based SDOclust architecture and running all experiments.

Includes:  
- fixed n-splits experiments  
- sequential / joblib / dask comparisons  
- baseline SDOclust comparison  

## How to Run
### Basic Run
```
python3 ./sdoclust_parallel.py
```

### Command line arguments
```
python3 sdoclust_parallel.py \
  --datasets blobs,noisy_blobs \
  --size 200000 \
  --backends seq,joblib,dask \
  --splits 2,4,8,16 \
  --seeds 42
```

***Arguments:***  

`--datasets`  Dataset types to evaluate.  
- blobs: clean Gaussian blobs  
- noisy_blobs: blobs with uniformly distributed noise points

`--size`  Number of samples: Multiple values can be provided as a comma separated list.

`--backends`  Label extension execution backend  
- seq: sequential  
- joblib: multi thread    
- dask: task-based parallel execution
  
`--splits`  Number of data splits.

`--seeds`  Random seeds for reproducibility.

## Results
```
######################################################################
DATASET=blobs  N=200000  d=10  centers=5  std=1.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.665s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.879   0.522   2.401    267  1.000  1.000
    seq      4  50000   0.967   0.568   1.535    267  1.000  1.000
    seq      8  25000   0.484   0.486   0.970    264  1.000  1.000
    seq     16  12500   0.290   0.466   0.756    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.888   0.497   2.384    267  1.000  1.000
 joblib      4  50000   0.915   0.294   1.209    267  1.000  1.000
 joblib      8  25000   0.467   0.175   0.642    264  1.000  1.000
 joblib     16  12500   0.301   0.132   0.433    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.849   0.704   2.553    267  1.000  1.000
   dask      4  50000   0.929   0.291   1.220    267  1.000  1.000
   dask      8  25000   0.460   0.155   0.616    264  1.000  1.000
   dask     16  12500   0.261   0.134   0.395    261  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.571s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.899   0.581   2.480    267  1.000  1.000
    seq      4  50000   0.960   0.558   1.517    263  1.000  1.000
    seq      8  25000   0.475   0.568   1.042    264  1.000  1.000
    seq     16  12500   0.312   0.602   0.914    258  1.000  0.999

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.864   0.588   2.452    267  1.000  1.000
 joblib      4  50000   0.942   0.292   1.233    263  1.000  1.000
 joblib      8  25000   0.476   0.183   0.659    264  1.000  1.000
 joblib     16  12500   0.301   0.161   0.462    258  1.000  0.999

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.862   0.578   2.441    267  1.000  1.000
   dask      4  50000   0.942   0.287   1.229    263  1.000  1.000
   dask      8  25000   0.463   0.175   0.638    264  1.000  1.000
   dask     16  12500   0.265   0.157   0.422    258  1.000  0.999

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.308s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.395   1.110   3.506    267  1.000  1.000
    seq      4  50000   1.264   1.076   2.341    266  1.000  1.000
    seq      8  25000   0.622   1.241   1.863    264  1.000  1.000
    seq     16  12500   0.355   1.228   1.584    263  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.519   1.101   3.620    267  1.000  1.000
 joblib      4  50000   1.226   0.560   1.786    266  1.000  1.000
 joblib      8  25000   0.605   0.352   0.957    264  1.000  1.000
 joblib     16  12500   0.331   0.298   0.629    263  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.397   1.080   3.477    267  1.000  1.000
   dask      4  50000   1.240   0.553   1.794    266  1.000  1.000
   dask      8  25000   0.613   0.350   0.963    264  1.000  1.000
   dask     16  12500   0.330   0.282   0.612    263  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.062s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.463   1.130   3.592    266  1.000  1.000
    seq      4  50000   1.228   1.113   2.341    266  1.000  1.000
    seq      8  25000   0.620   1.113   1.732    265  1.000  1.000
    seq     16  12500   0.332   1.112   1.444    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.410   1.175   3.586    266  1.000  1.000
 joblib      4  50000   1.268   0.581   1.849    266  1.000  1.000
 joblib      8  25000   0.618   0.348   0.966    265  1.000  1.000
 joblib     16  12500   0.337   0.294   0.631    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.397   1.127   3.525    266  1.000  1.000
   dask      4  50000   1.267   0.572   1.839    266  1.000  1.000
   dask      8  25000   0.606   0.348   0.954    265  1.000  1.000
   dask     16  12500   0.336   0.278   0.614    261  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.536s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.916   0.511   2.427    266  1.000  1.000
    seq      4  52500   0.998   0.507   1.505    267  1.000  1.000
    seq      8  26250   0.487   0.567   1.053    264  1.000  1.000
    seq     16  13125   0.307   0.573   0.880    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.937   0.518   2.455    266  1.000  1.000
 joblib      4  52500   0.933   0.266   1.198    267  1.000  1.000
 joblib      8  26250   0.492   0.184   0.676    264  1.000  1.000
 joblib     16  13125   0.277   0.153   0.430    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.935   0.513   2.448    266  1.000  1.000
   dask      4  52500   0.982   0.261   1.243    267  1.000  1.000
   dask      8  26250   0.509   0.182   0.691    264  1.000  1.000
   dask     16  13125   0.272   0.152   0.425    261  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.612s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.949   0.557   2.506    265  1.000  1.000
    seq      4  52500   1.000   0.518   1.518    267  1.000  1.000
    seq      8  26250   0.498   0.596   1.094    264  1.000  1.000
    seq     16  13125   0.301   0.548   0.849    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.956   0.566   2.522    265  1.000  1.000
 joblib      4  52500   0.980   0.269   1.249    267  1.000  1.000
 joblib      8  26250   0.515   0.187   0.702    264  1.000  1.000
 joblib     16  13125   0.266   0.151   0.417    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.951   0.558   2.509    265  1.000  1.000
   dask      4  52500   0.994   0.268   1.261    267  1.000  1.000
   dask      8  26250   0.494   0.187   0.680    264  1.000  1.000
   dask     16  13125   0.268   0.146   0.415    261  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.930s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.650   1.104   3.754    267  1.000  1.000
    seq      4  52500   1.295   1.157   2.452    266  1.000  1.000
    seq      8  26250   0.675   1.204   1.879    262  1.000  1.000
    seq     16  13125   0.359   1.136   1.495    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.661   1.177   3.838    267  1.000  1.000
 joblib      4  52500   1.267   0.601   1.867    266  1.000  1.000
 joblib      8  26250   0.621   0.364   0.985    262  1.000  1.000
 joblib     16  13125   0.427   0.342   0.768    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.723   1.300   4.023    267  1.000  1.000
   dask      4  52500   1.331   0.598   1.929    266  1.000  1.000
   dask      8  26250   0.634   0.371   1.006    262  1.000  1.000
   dask     16  13125   0.354   0.287   0.642    261  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.861s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.616   1.215   3.831    263  1.000  1.000
    seq      4  52500   1.347   1.190   2.537    265  1.000  1.000
    seq      8  26250   0.641   1.224   1.865    263  1.000  1.000
    seq     16  13125   0.341   1.346   1.687    260  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.778   1.275   4.053    263  1.000  1.000
 joblib      4  52500   1.444   0.615   2.059    265  1.000  1.000
 joblib      8  26250   0.645   0.384   1.029    263  1.000  1.000
 joblib     16  13125   0.375   0.288   0.664    260  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.526   1.178   3.704    263  1.000  1.000
   dask      4  52500   1.355   0.624   1.978    265  1.000  1.000
   dask      8  26250   0.632   0.389   1.021    263  1.000  1.000
   dask     16  13125   0.343   0.292   0.635    260  1.000  1.000
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
