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
- KNN neighbors for label extension: Determined internally by SDOclust on splitA  
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
Baseline SDOclust Time=4.363s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.093s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.134s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=1.811s  ARI=1.000  AMI=1.000
MiniBatchKMeans Dask      Time=0.664s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.743   0.533   2.276    267  1.000  1.000
    seq      4  50000   0.858   0.528   1.385    267  0.993  0.992
    seq      8  25000   0.400   0.549   0.949    264  1.000  1.000
    seq     16  12500   0.212   0.524   0.736    261  1.000  1.000

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.751   0.537   2.288    267  1.000  1.000
 joblib      4  50000   0.856   0.279   1.136    267  0.993  0.992
 joblib      8  25000   0.405   0.178   0.582    264  1.000  1.000
 joblib     16  12500   0.207   0.145   0.352    261  1.000  1.000

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.761   0.528   2.289    267  1.000  1.000
   dask      4  50000   0.836   0.273   1.109    267  0.993  0.992
   dask      8  25000   0.402   0.175   0.577    264  1.000  1.000
   dask     16  12500   0.204   0.139   0.343    261  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.489s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.042s  ARI=1.000  AMI=0.999
MiniBatchKMeans Seq       Time=0.081s  ARI=1.000  AMI=0.999
MiniBatchKMeans Joblib    Time=0.179s  ARI=0.719  AMI=0.865
MiniBatchKMeans Dask      Time=0.395s  ARI=0.719  AMI=0.865

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   1.762   0.680   2.442    267  1.000  1.000
    seq      4  50000   0.852   0.672   1.524    263  1.000  0.999
    seq      8  25000   0.402   0.654   1.056    264  1.000  1.000
    seq     16  12500   0.206   0.668   0.874    258  1.000  1.000

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   1.749   0.692   2.441    267  1.000  1.000
 joblib      4  50000   0.858   0.355   1.213    263  1.000  0.999
 joblib      8  25000   0.402   0.215   0.617    264  1.000  1.000
 joblib     16  12500   0.209   0.183   0.392    258  1.000  1.000

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   1.731   0.682   2.413    267  1.000  1.000
   dask      4  50000   0.862   0.344   1.206    263  1.000  0.999
   dask      8  25000   0.406   0.201   0.607    264  1.000  1.000
   dask     16  12500   0.209   0.169   0.379    258  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=5.939s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.097s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.028s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=0.943s  ARI=1.000  AMI=1.000
MiniBatchKMeans Dask      Time=1.100s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.331   0.351   2.682    267  1.000  1.000
    seq      4  50000   1.158   0.345   1.503    266  1.000  1.000
    seq      8  25000   0.537   0.344   0.881    264  1.000  1.000
    seq     16  12500   0.275   0.356   0.631    263  1.000  1.000

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.315   0.379   2.695    267  1.000  1.000
 joblib      4  50000   1.140   0.249   1.389    266  1.000  1.000
 joblib      8  25000   0.542   0.239   0.781    264  1.000  1.000
 joblib     16  12500   0.274   0.235   0.509    263  1.000  1.000

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.318   0.381   2.699    267  1.000  1.000
   dask      4  50000   1.142   0.245   1.386    266  1.000  1.000
   dask      8  25000   0.539   0.227   0.766    264  1.000  1.000
   dask     16  12500   0.275   0.235   0.510    263  1.000  1.000

######################################################################
DATASET=blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.083s  ARI=1.000  AMI=1.000
Baseline parallel kmeans
KMeans Sequential         Time=0.085s  ARI=1.000  AMI=1.000
MiniBatchKMeans Seq       Time=0.030s  ARI=1.000  AMI=1.000
MiniBatchKMeans Joblib    Time=0.315s  ARI=0.781  AMI=0.904
MiniBatchKMeans Dask      Time=0.468s  ARI=0.781  AMI=0.904

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 100000   2.303   0.353   2.656    266  1.000  1.000
    seq      4  50000   1.147   0.335   1.483    266  1.000  1.000
    seq      8  25000   0.539   0.332   0.871    265  1.000  1.000
    seq     16  12500   0.275   0.349   0.624    261  1.000  1.000

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 100000   2.317   0.394   2.711    266  1.000  1.000
 joblib      4  50000   1.146   0.257   1.403    266  1.000  1.000
 joblib      8  25000   0.540   0.236   0.777    265  1.000  1.000
 joblib     16  12500   0.275   0.232   0.507    261  1.000  1.000

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 100000   2.311   0.375   2.685    266  1.000  1.000
   dask      4  50000   1.145   0.250   1.395    266  1.000  1.000
   dask      8  25000   0.542   0.237   0.779    265  1.000  1.000
   dask     16  12500   0.278   0.223   0.501    261  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.408s  ARI=0.936  AMI=0.921
Baseline parallel kmeans
KMeans Sequential         Time=0.046s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.059s  ARI=0.936  AMI=0.920
MiniBatchKMeans Joblib    Time=0.163s  ARI=0.748  AMI=0.854
MiniBatchKMeans Dask      Time=0.391s  ARI=0.748  AMI=0.854

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.866   0.601   2.467    266  0.936  0.921
    seq      4  52500   0.918   0.612   1.531    267  0.936  0.920
    seq      8  26250   0.434   0.609   1.043    264  0.936  0.921
    seq     16  13125   0.230   0.591   0.821    261  0.936  0.920

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.846   0.614   2.460    266  0.936  0.921
 joblib      4  52500   0.902   0.327   1.229    267  0.936  0.920
 joblib      8  26250   0.428   0.197   0.625    264  0.936  0.921
 joblib     16  13125   0.214   0.150   0.364    261  0.936  0.920

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.943   0.604   2.547    266  0.936  0.921
   dask      4  52500   0.907   0.315   1.222    267  0.936  0.920
   dask      8  26250   0.418   0.194   0.613    264  0.936  0.921
   dask     16  13125   0.214   0.152   0.366    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=10  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=4.729s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.070s  ARI=0.672  AMI=0.793
MiniBatchKMeans Seq       Time=0.049s  ARI=0.670  AMI=0.793
MiniBatchKMeans Joblib    Time=0.122s  ARI=0.744  AMI=0.843
MiniBatchKMeans Dask      Time=0.343s  ARI=0.744  AMI=0.843

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   1.860   0.670   2.530    265  0.936  0.920
    seq      4  52500   0.923   0.735   1.658    267  0.936  0.920
    seq      8  26250   0.439   0.737   1.176    264  0.936  0.920
    seq     16  13125   0.234   0.722   0.957    261  0.936  0.920

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   1.864   0.678   2.542    265  0.936  0.920
 joblib      4  52500   0.902   0.380   1.282    267  0.936  0.920
 joblib      8  26250   0.431   0.252   0.683    264  0.936  0.920
 joblib     16  13125   0.216   0.189   0.405    261  0.936  0.920

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   1.861   0.674   2.535    265  0.936  0.920
   dask      4  52500   0.907   0.380   1.286    267  0.936  0.920
   dask      8  26250   0.423   0.226   0.649    264  0.936  0.920
   dask     16  13125   0.216   0.182   0.398    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.360s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.130s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.096s  ARI=0.936  AMI=0.920
MiniBatchKMeans Joblib    Time=0.207s  ARI=0.623  AMI=0.802
MiniBatchKMeans Dask      Time=0.355s  ARI=0.623  AMI=0.802

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.425   0.359   2.784    267  0.936  0.920
    seq      4  52500   1.203   0.362   1.565    266  0.936  0.920
    seq      8  26250   0.577   0.355   0.932    262  0.936  0.920
    seq     16  13125   0.287   0.381   0.668    261  0.936  0.920

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.427   0.387   2.814    267  0.936  0.920
 joblib      4  52500   1.187   0.271   1.459    266  0.936  0.920
 joblib      8  26250   0.566   0.256   0.822    262  0.936  0.920
 joblib     16  13125   0.287   0.242   0.530    261  0.936  0.920

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.413   0.391   2.805    267  0.936  0.920
   dask      4  52500   1.188   0.250   1.438    266  0.936  0.920
   dask      8  26250   0.563   0.242   0.805    262  0.936  0.920
   dask     16  13125   0.288   0.232   0.520    261  0.936  0.920

######################################################################
DATASET=noisy_blobs  N=200000  d=50  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=6.256s  ARI=0.936  AMI=0.920
Baseline parallel kmeans
KMeans Sequential         Time=0.099s  ARI=0.936  AMI=0.920
MiniBatchKMeans Seq       Time=0.104s  ARI=0.780  AMI=0.898
MiniBatchKMeans Joblib    Time=0.207s  ARI=0.615  AMI=0.787
MiniBatchKMeans Dask      Time=0.365s  ARI=0.615  AMI=0.787

======================================================================
[backend=seq] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2 105000   2.439   0.353   2.793    263  0.936  0.920
    seq      4  52500   1.200   0.352   1.553    265  0.936  0.920
    seq      8  26250   0.565   0.361   0.926    263  0.936  0.920
    seq     16  13125   0.288   0.361   0.649    260  0.936  0.920

======================================================================
[backend=joblib] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2 105000   2.429   0.383   2.812    263  0.936  0.920
 joblib      4  52500   1.228   0.262   1.490    265  0.936  0.920
 joblib      8  26250   0.567   0.266   0.833    263  0.936  0.920
 joblib     16  13125   0.290   0.238   0.528    260  0.936  0.920

======================================================================
[backend=dask] chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2 105000   2.426   0.398   2.824    263  0.936  0.920
   dask      4  52500   1.211   0.253   1.463    265  0.936  0.920
   dask      8  26250   0.566   0.238   0.803    263  0.936  0.920
   dask     16  13125   0.288   0.257   0.545    260  0.936  0.920
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
