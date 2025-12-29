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

Baselines:  
- Full SDOclust executed on the entire dataset (no splitting), serving as an algorithmic baseline.  
- Established clustering baselines including scikit-learn KMeans and MiniBatchKMeans, as well as a parallel MiniBatchKMeans implementation using joblib and dask.  


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
Baseline SDOclust Time=4.650s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.135s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.103s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=1.895s  ARI=1.000  AMI=1.000
MiniBatchKMeans Dask      Time=0.666s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.806   0.494   2.299    267  1.000  1.000
    seq      4  50000   0.884   0.585   1.469    267  1.000  1.000
    seq      8  25000   0.416   0.490   0.906    264  1.000  1.000
    seq     16  12500   0.223   0.468   0.691    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.833   0.497   2.330    267  1.000  1.000
 joblib      4  50000   0.859   0.363   1.222    267  1.000  1.000
 joblib      8  25000   0.540   0.169   0.709    264  1.000  1.000
 joblib     16  12500   0.220   0.134   0.354    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.794   0.499   2.294    267  1.000  1.000
   dask      4  50000   0.856   0.315   1.171    267  1.000  1.000
   dask      8  25000   0.408   0.158   0.566    264  1.000  1.000
   dask     16  12500   0.207   0.123   0.330    261  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.704s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.044s  ARI=1.000  AMI=0.999
MiniBatchKMeans Seq       Time=0.080s  ARI=1.000  AMI=0.999
MiniBatchKMeans Joblib    Time=0.182s  ARI=0.719  AMI=0.865
MiniBatchKMeans Dask      Time=0.398s  ARI=0.719  AMI=0.865

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.757   0.578   2.336    267  1.000  1.000
    seq      4  50000   0.859   0.560   1.419    263  1.000  1.000
    seq      8  25000   0.406   0.557   0.963    264  1.000  1.000
    seq     16  12500   0.210   0.596   0.806    258  1.000  0.999

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.767   0.607   2.373    267  1.000  1.000
 joblib      4  50000   0.875   0.298   1.174    263  1.000  1.000
 joblib      8  25000   0.405   0.179   0.583    264  1.000  1.000
 joblib     16  12500   0.213   0.156   0.369    258  1.000  0.999

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.751   0.579   2.330    267  1.000  1.000
   dask      4  50000   0.862   0.287   1.149    263  1.000  1.000
   dask      8  25000   0.405   0.167   0.571    264  1.000  1.000
   dask     16  12500   0.210   0.152   0.362    258  1.000  0.999

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.514s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.123s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.029s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=0.915s  ARI=1.000  AMI=1.000
MiniBatchKMeans Dask      Time=1.101s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.347   1.137   3.485    267  1.000  1.000
    seq      4  50000   1.168   1.095   2.263    266  1.000  1.000
    seq      8  25000   0.545   1.139   1.684    264  1.000  1.000
    seq     16  12500   0.274   1.212   1.486    263  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.449   1.157   3.606    267  1.000  1.000
 joblib      4  50000   1.169   0.591   1.760    266  1.000  1.000
 joblib      8  25000   0.554   0.362   0.916    264  1.000  1.000
 joblib     16  12500   0.269   0.291   0.560    263  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.453   1.112   3.566    267  1.000  1.000
   dask      4  50000   1.196   0.567   1.763    266  1.000  1.000
   dask      8  25000   0.574   0.372   0.946    264  1.000  1.000
   dask     16  12500   0.273   0.291   0.564    263  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.576s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.110s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.028s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=0.329s  ARI=0.781  AMI=0.904
MiniBatchKMeans Dask      Time=0.464s  ARI=0.781  AMI=0.904

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.319   1.141   3.460    266  1.000  1.000
    seq      4  50000   1.150   1.113   2.264    266  1.000  1.000
    seq      8  25000   0.538   1.116   1.654    265  1.000  1.000
    seq     16  12500   0.273   1.116   1.390    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.314   1.154   3.468    266  1.000  1.000
 joblib      4  50000   1.146   0.580   1.725    266  1.000  1.000
 joblib      8  25000   0.533   0.355   0.889    265  1.000  1.000
 joblib     16  12500   0.271   0.286   0.557    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.309   1.135   3.445    266  1.000  1.000
   dask      4  50000   1.138   0.572   1.709    266  1.000  1.000
   dask      8  25000   0.536   0.348   0.884    265  1.000  1.000
   dask     16  12500   0.269   0.297   0.566    261  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.692s  ARI=0.936  AMI=0.921
Baseline parallel kmeans
KMeans Sequential         Time=0.046s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.059s  ARI=0.936  AMI=0.920
MiniBatchKMeans Joblib    Time=0.175s  ARI=0.748  AMI=0.854
MiniBatchKMeans Dask      Time=0.393s  ARI=0.748  AMI=0.854

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.830   0.512   2.341    266  0.936  0.921
    seq      4  52500   0.909   0.508   1.416    267  0.936  0.921
    seq      8  26250   0.444   0.572   1.016    264  0.936  0.921
    seq     16  13125   0.221   0.575   0.796    261  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.821   0.517   2.338    266  0.936  0.921
 joblib      4  52500   0.952   0.266   1.218    267  0.936  0.921
 joblib      8  26250   0.417   0.193   0.610    264  0.936  0.921
 joblib     16  13125   0.220   0.150   0.369    261  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.848   0.511   2.358    266  0.936  0.921
   dask      4  52500   0.895   0.261   1.155    267  0.936  0.921
   dask      8  26250   0.424   0.180   0.604    264  0.936  0.921
   dask     16  13125   0.213   0.148   0.362    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.598s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.063s  ARI=0.672  AMI=0.793
MiniBatchKMeans Seq       Time=0.044s  ARI=0.670  AMI=0.793
MiniBatchKMeans Joblib    Time=0.115s  ARI=0.744  AMI=0.843
MiniBatchKMeans Dask      Time=0.348s  ARI=0.744  AMI=0.843

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.888   0.582   2.470    265  0.936  0.920
    seq      4  52500   0.924   0.522   1.446    267  0.936  0.920
    seq      8  26250   0.441   0.642   1.083    264  0.936  0.920
    seq     16  13125   0.239   0.551   0.790    261  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.864   0.566   2.430    265  0.936  0.920
 joblib      4  52500   0.903   0.282   1.185    267  0.936  0.920
 joblib      8  26250   0.433   0.196   0.629    264  0.936  0.920
 joblib     16  13125   0.217   0.146   0.363    261  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.850   0.559   2.409    265  0.936  0.920
   dask      4  52500   0.907   0.267   1.174    267  0.936  0.920
   dask      8  26250   0.425   0.197   0.622    264  0.936  0.920
   dask     16  13125   0.216   0.145   0.361    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.744s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.125s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.106s  ARI=0.936  AMI=0.920
MiniBatchKMeans Joblib    Time=0.227s  ARI=0.623  AMI=0.802
MiniBatchKMeans Dask      Time=0.367s  ARI=0.623  AMI=0.802

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.485   1.106   3.591    267  0.936  0.920
    seq      4  52500   1.210   1.210   2.420    266  0.936  0.920
    seq      8  26250   0.570   1.178   1.749    262  0.936  0.920
    seq     16  13125   0.291   1.160   1.451    261  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.426   1.119   3.545    267  0.936  0.920
 joblib      4  52500   1.206   0.607   1.812    266  0.936  0.920
 joblib      8  26250   0.559   0.369   0.928    262  0.936  0.920
 joblib     16  13125   0.286   0.302   0.588    261  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.432   1.115   3.547    267  0.936  0.920
   dask      4  52500   1.204   0.604   1.808    266  0.936  0.920
   dask      8  26250   0.562   0.365   0.928    262  0.936  0.920
   dask     16  13125   0.286   0.299   0.585    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.600s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.127s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.110s  ARI=0.780  AMI=0.898
MiniBatchKMeans Joblib    Time=0.209s  ARI=0.615  AMI=0.787
MiniBatchKMeans Dask      Time=0.375s  ARI=0.615  AMI=0.787

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.439   1.190   3.629    263  0.936  0.920
    seq      4  52500   1.247   1.204   2.451    265  0.936  0.920
    seq      8  26250   0.576   1.242   1.818    263  0.936  0.920
    seq     16  13125   0.293   1.165   1.458    260  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.437   1.190   3.626    263  0.936  0.920
 joblib      4  52500   1.206   0.626   1.832    265  0.936  0.920
 joblib      8  26250   0.566   0.382   0.948    263  0.936  0.920
 joblib     16  13125   0.287   0.308   0.595    260  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.439   1.192   3.632    263  0.936  0.920
   dask      4  52500   1.210   0.618   1.828    265  0.936  0.920
   dask      8  26250   0.565   0.381   0.947    263  0.936  0.920
   dask     16  13125   0.286   0.297   0.583    260  0.936  0.920
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
