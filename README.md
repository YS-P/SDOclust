# SDOclust

## Overview
This project aims to develop a parallelized version of SDOclust and evaluate its performance against baseline SDOclust and established clustering methods.

## Methodology
1. Data Splitting  
The dataset is divided into multiple splits of similar size.
  
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

Results are reported in tables:  
- Scalability with respect split size  
- Performance differences between execution backends  
- Trade offs between computation time and clustering quality

Baselines:  
- Baselines include full SDOclust executed on the entire dataset, and scikit-learn KMeans and MiniBatchKMeans with sequential, joblib, and dask backends.


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
Baseline SDOclust Time=4.400s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.095s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.140s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=1.882s  ARI=1.000  AMI=1.000
MiniBatchKMeans Dask      Time=0.631s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.782   0.526   2.309    267  1.000  1.000
    seq      4  50000   0.871   0.521   1.392    267  1.000  1.000
    seq      8  25000   0.415   0.537   0.952    264  1.000  1.000
    seq     16  12500   0.221   0.517   0.738    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.756   0.524   2.279    267  1.000  1.000
 joblib      4  50000   0.852   0.282   1.134    267  1.000  1.000
 joblib      8  25000   0.409   0.179   0.588    264  1.000  1.000
 joblib     16  12500   0.212   0.148   0.360    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.753   0.521   2.275    267  1.000  1.000
   dask      4  50000   0.847   0.272   1.119    267  1.000  1.000
   dask      8  25000   0.400   0.169   0.569    264  1.000  1.000
   dask     16  12500   0.208   0.136   0.344    261  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.483s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.044s  ARI=1.000  AMI=0.999
MiniBatchKMeans Seq       Time=0.083s  ARI=1.000  AMI=0.999
MiniBatchKMeans Joblib    Time=0.188s  ARI=0.719  AMI=0.865
MiniBatchKMeans Dask      Time=0.398s  ARI=0.719  AMI=0.865

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.751   0.633   2.385    267  1.000  1.000
    seq      4  50000   0.863   0.628   1.491    263  1.000  1.000
    seq      8  25000   0.424   0.623   1.048    264  1.000  1.000
    seq     16  12500   0.211   0.624   0.835    258  1.000  0.999

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.766   0.638   2.404    267  1.000  1.000
 joblib      4  50000   0.869   0.335   1.204    263  1.000  1.000
 joblib      8  25000   0.405   0.199   0.604    264  1.000  1.000
 joblib     16  12500   0.210   0.162   0.372    258  1.000  0.999

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.756   0.630   2.385    267  1.000  1.000
   dask      4  50000   0.863   0.328   1.191    263  1.000  1.000
   dask      8  25000   0.405   0.192   0.597    264  1.000  1.000
   dask     16  12500   0.210   0.162   0.372    258  1.000  0.999

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=5.915s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.096s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.027s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=0.933s  ARI=1.000  AMI=1.000
MiniBatchKMeans Dask      Time=1.075s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.310   0.369   2.679    267  1.000  1.000
    seq      4  50000   1.158   0.353   1.511    266  1.000  1.000
    seq      8  25000   0.536   0.369   0.906    264  1.000  1.000
    seq     16  12500   0.273   0.374   0.647    263  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.315   0.407   2.722    267  1.000  1.000
 joblib      4  50000   1.138   0.265   1.404    266  1.000  1.000
 joblib      8  25000   0.537   0.217   0.754    264  1.000  1.000
 joblib     16  12500   0.281   0.206   0.487    263  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.338   0.398   2.736    267  1.000  1.000
   dask      4  50000   1.169   0.239   1.407    266  1.000  1.000
   dask      8  25000   0.538   0.205   0.742    264  1.000  1.000
   dask     16  12500   0.279   0.194   0.474    263  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=5.999s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.085s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.028s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=0.309s  ARI=0.781  AMI=0.904
MiniBatchKMeans Dask      Time=0.473s  ARI=0.781  AMI=0.904

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.324   0.383   2.707    266  1.000  1.000
    seq      4  50000   1.199   0.358   1.557    266  1.000  1.000
    seq      8  25000   0.540   0.369   0.909    265  1.000  1.000
    seq     16  12500   0.277   0.368   0.645    261  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.400   0.392   2.792    266  1.000  1.000
 joblib      4  50000   1.154   0.251   1.405    266  1.000  1.000
 joblib      8  25000   0.541   0.221   0.762    265  1.000  1.000
 joblib     16  12500   0.275   0.215   0.490    261  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.337   0.370   2.706    266  1.000  1.000
   dask      4  50000   1.153   0.240   1.393    266  1.000  1.000
   dask      8  25000   0.543   0.214   0.757    265  1.000  1.000
   dask     16  12500   0.274   0.202   0.476    261  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.602s  ARI=0.936  AMI=0.921
Baseline parallel kmeans
KMeans Sequential         Time=0.048s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.059s  ARI=0.936  AMI=0.920
MiniBatchKMeans Joblib    Time=0.180s  ARI=0.748  AMI=0.854
MiniBatchKMeans Dask      Time=0.392s  ARI=0.748  AMI=0.854

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.869   0.597   2.466    266  0.936  0.921
    seq      4  52500   0.922   0.611   1.533    267  0.936  0.921
    seq      8  26250   0.437   0.592   1.028    264  0.936  0.921
    seq     16  13125   0.232   0.582   0.814    261  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.844   0.597   2.442    266  0.936  0.921
 joblib      4  52500   0.909   0.306   1.216    267  0.936  0.921
 joblib      8  26250   0.431   0.185   0.616    264  0.936  0.921
 joblib     16  13125   0.215   0.157   0.371    261  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.861   0.625   2.486    266  0.936  0.921
   dask      4  52500   0.902   0.304   1.206    267  0.936  0.921
   dask      8  26250   0.421   0.187   0.608    264  0.936  0.921
   dask     16  13125   0.212   0.153   0.366    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.574s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.068s  ARI=0.672  AMI=0.793
MiniBatchKMeans Seq       Time=0.051s  ARI=0.670  AMI=0.793
MiniBatchKMeans Joblib    Time=0.134s  ARI=0.744  AMI=0.843
MiniBatchKMeans Dask      Time=0.348s  ARI=0.744  AMI=0.843

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.871   0.644   2.515    265  0.936  0.920
    seq      4  52500   0.932   0.677   1.609    267  0.936  0.920
    seq      8  26250   0.439   0.677   1.115    264  0.936  0.920
    seq     16  13125   0.226   0.664   0.890    261  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.857   0.640   2.497    265  0.936  0.920
 joblib      4  52500   0.909   0.359   1.267    267  0.936  0.920
 joblib      8  26250   0.432   0.220   0.652    264  0.936  0.920
 joblib     16  13125   0.216   0.175   0.391    261  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.865   0.635   2.501    265  0.936  0.920
   dask      4  52500   0.904   0.350   1.254    267  0.936  0.920
   dask      8  26250   0.423   0.211   0.634    264  0.936  0.920
   dask     16  13125   0.216   0.169   0.385    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.229s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.104s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.093s  ARI=0.936  AMI=0.920
MiniBatchKMeans Joblib    Time=0.207s  ARI=0.623  AMI=0.802
MiniBatchKMeans Dask      Time=0.361s  ARI=0.623  AMI=0.802

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.441   0.381   2.822    267  0.936  0.920
    seq      4  52500   1.196   0.386   1.582    266  0.936  0.920
    seq      8  26250   0.563   0.390   0.953    262  0.936  0.920
    seq     16  13125   0.286   0.395   0.682    261  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.410   0.417   2.827    267  0.936  0.920
 joblib      4  52500   1.193   0.258   1.451    266  0.936  0.920
 joblib      8  26250   0.563   0.215   0.778    262  0.936  0.920
 joblib     16  13125   0.288   0.203   0.491    261  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.449   0.432   2.880    267  0.936  0.920
   dask      4  52500   1.255   0.274   1.529    266  0.936  0.920
   dask      8  26250   0.653   0.222   0.875    262  0.936  0.920
   dask     16  13125   0.338   0.241   0.580    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.532s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.102s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.101s  ARI=0.780  AMI=0.898
MiniBatchKMeans Joblib    Time=0.222s  ARI=0.615  AMI=0.787
MiniBatchKMeans Dask      Time=0.375s  ARI=0.615  AMI=0.787

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.430   0.371   2.800    263  0.936  0.920
    seq      4  52500   1.199   0.371   1.569    265  0.936  0.920
    seq      8  26250   0.580   0.384   0.964    263  0.936  0.920
    seq     16  13125   0.290   0.392   0.683    260  0.936  0.920

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.416   0.409   2.825    263  0.936  0.920
 joblib      4  52500   1.216   0.252   1.468    265  0.936  0.920
 joblib      8  26250   0.564   0.221   0.785    263  0.936  0.920
 joblib     16  13125   0.288   0.284   0.572    260  0.936  0.920

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.430   0.405   2.835    263  0.936  0.920
   dask      4  52500   1.199   0.266   1.465    265  0.936  0.920
   dask      8  26250   0.566   0.225   0.791    263  0.936  0.920
```
-	Reducing the size of split A significantly decreases SDOclust fitting time while maintaining comparable clustering results.  
-	Joblib and Dask reduce label extension time compared to sequential execution.  
-	Joblib and Dask show similar runtime and identical clustering quality.  
-	Compared to baseline SDOclust, the proposed approach achieves similar clustering quality with significantly reduced total runtime.  
-	Overall runtime decreases as the number of splits increases.  
-	Performance is stable across different cluster standard deviations, degrades slightly with noise, and is more efficient in lower dimensions.  

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
