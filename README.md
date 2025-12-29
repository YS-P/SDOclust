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
DATASET=blobs  N=50000  d=10  centers=5  std=1.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.106s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  25000   0.502   0.141   0.643    263  1.000  1.000
    seq      4  12500   0.260   0.124   0.384    259  1.000  1.000
    seq      8   6250   0.151   0.160   0.311    255  1.000  1.000
    seq     16   3125   0.080   0.132   0.211    243  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  25000   0.460   0.144   0.605    263  1.000  1.000
 joblib      4  12500   0.274   0.069   0.344    259  1.000  1.000
 joblib      8   6250   0.127   0.123   0.250    255  1.000  1.000
 joblib     16   3125   0.085   0.078   0.163    243  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  25000   0.442   0.345   0.787    263  1.000  1.000
   dask      4  12500   0.268   0.066   0.334    259  1.000  1.000
   dask      8   6250   0.155   0.117   0.272    255  1.000  1.000
   dask     16   3125   0.082   0.095   0.177    243  1.000  1.000

######################################################################
DATASET=blobs  N=50000  d=10  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.208s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  25000   0.484   0.151   0.635    262  1.000  1.000
    seq      4  12500   0.267   0.147   0.414    260  1.000  0.999
    seq      8   6250   0.142   0.184   0.326    255  1.000  1.000
    seq     16   3125   0.078   0.176   0.254    240  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  25000   0.460   0.150   0.610    262  1.000  1.000
 joblib      4  12500   0.261   0.079   0.340    260  1.000  0.999
 joblib      8   6250   0.138   0.100   0.238    255  1.000  1.000
 joblib     16   3125   0.075   0.097   0.172    240  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  25000   0.483   0.144   0.627    262  1.000  1.000
   dask      4  12500   0.262   0.079   0.341    260  1.000  0.999
   dask      8   6250   0.161   0.082   0.243    255  1.000  1.000
   dask     16   3125   0.073   0.082   0.155    240  1.000  1.000

######################################################################
DATASET=blobs  N=50000  d=50  centers=5  std=1.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.502s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  25000   0.636   0.266   0.902    263  1.000  1.000
    seq      4  12500   0.346   0.266   0.612    261  1.000  1.000
    seq      8   6250   0.182   0.317   0.499    253  1.000  1.000
    seq     16   3125   0.103   0.349   0.452    239  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  25000   0.665   0.273   0.938    263  1.000  1.000
 joblib      4  12500   0.350   0.143   0.493    261  1.000  1.000
 joblib      8   6250   0.173   0.136   0.309    253  1.000  1.000
 joblib     16   3125   0.085   0.181   0.266    239  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  25000   0.622   0.267   0.889    263  1.000  1.000
   dask      4  12500   0.325   0.139   0.464    261  1.000  1.000
   dask      8   6250   0.172   0.148   0.321    253  1.000  1.000
   dask     16   3125   0.114   0.139   0.253    239  1.000  1.000

######################################################################
DATASET=blobs  N=50000  d=50  centers=5  std=2.0  noise_frac=0.0
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.485s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  25000   0.607   0.272   0.879    263  1.000  1.000
    seq      4  12500   0.321   0.275   0.596    258  1.000  1.000
    seq      8   6250   0.170   0.272   0.442    258  1.000  1.000
    seq     16   3125   0.087   0.324   0.411    237  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  25000   0.614   0.275   0.889    263  1.000  1.000
 joblib      4  12500   0.328   0.144   0.472    258  1.000  1.000
 joblib      8   6250   0.197   0.116   0.313    258  1.000  1.000
 joblib     16   3125   0.087   0.141   0.228    237  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  25000   0.611   0.273   0.884    263  1.000  1.000
   dask      4  12500   0.328   0.143   0.471    258  1.000  1.000
   dask      8   6250   0.176   0.141   0.317    258  1.000  1.000
   dask     16   3125   0.087   0.156   0.242    237  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=50000  d=10  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.180s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  26250   0.488   0.138   0.626    264  1.000  1.000
    seq      4  13125   0.288   0.140   0.428    261  1.000  1.000
    seq      8   6563   0.176   0.158   0.334    253  1.000  1.000
    seq     16   3282   0.092   0.202   0.294    241  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  26250   0.510   0.146   0.656    264  1.000  1.000
 joblib      4  13125   0.279   0.089   0.368    261  1.000  1.000
 joblib      8   6563   0.135   0.114   0.250    253  1.000  1.000
 joblib     16   3282   0.067   0.110   0.177    241  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  26250   0.485   0.139   0.624    264  1.000  1.000
   dask      4  13125   0.267   0.078   0.345    261  1.000  1.000
   dask      8   6563   0.135   0.097   0.232    253  1.000  1.000
   dask     16   3282   0.068   0.077   0.145    241  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=50000  d=10  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.156s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  26250   0.530   0.148   0.678    264  1.000  1.000
    seq      4  13125   0.295   0.147   0.442    263  1.000  1.000
    seq      8   6563   0.192   0.186   0.378    254  1.000  1.000
    seq     16   3282   0.079   0.206   0.285    246  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  26250   0.488   0.149   0.638    264  1.000  1.000
 joblib      4  13125   0.279   0.088   0.368    263  1.000  1.000
 joblib      8   6563   0.149   0.115   0.264    254  1.000  1.000
 joblib     16   3282   0.076   0.103   0.179    246  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  26250   0.497   0.148   0.645    264  1.000  1.000
   dask      4  13125   0.276   0.078   0.354    263  1.000  1.000
   dask      8   6563   0.159   0.122   0.282    254  1.000  1.000
   dask     16   3282   0.081   0.087   0.168    246  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=50000  d=50  centers=5  std=1.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.585s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  26250   0.632   0.295   0.927    263  1.000  1.000
    seq      4  13125   0.361   0.275   0.636    263  1.000  1.000
    seq      8   6563   0.174   0.301   0.475    252  1.000  1.000
    seq     16   3282   0.094   0.351   0.445    248  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  26250   0.656   0.312   0.968    263  1.000  1.000
 joblib      4  13125   0.348   0.156   0.504    263  1.000  1.000
 joblib      8   6563   0.186   0.133   0.320    252  1.000  1.000
 joblib     16   3282   0.104   0.170   0.273    248  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  26250   0.655   0.298   0.953    263  1.000  1.000
   dask      4  13125   0.370   0.143   0.513    263  1.000  1.000
   dask      8   6563   0.211   0.124   0.335    252  1.000  1.000
   dask     16   3282   0.085   0.140   0.224    248  1.000  1.000

######################################################################
DATASET=noisy_blobs  N=50000  d=50  centers=5  std=2.0  noise_frac=0.05
######################################################################
[SEED=42]

----------------------------------------------------------------------
Baseline SDOclust Time=1.655s  ARI=1.000  AMI=1.000

======================================================================
[backend=seq] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
    seq      2  26250   0.653   0.312   0.965    264  1.000  1.000
    seq      4  13125   0.355   0.286   0.641    262  1.000  1.000
    seq      8   6563   0.169   0.320   0.490    257  1.000  1.000
    seq     16   3282   0.104   0.359   0.462    240  1.000  1.000

======================================================================
[backend=joblib] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
 joblib      2  26250   0.653   0.323   0.976    264  1.000  1.000
 joblib      4  13125   0.367   0.159   0.527    262  1.000  1.000
 joblib      8   6563   0.209   0.141   0.351    257  1.000  1.000
 joblib     16   3282   0.089   0.116   0.205    240  1.000  1.000

======================================================================
[backend=dask] knn=10 chunksize=2000
backend splitA    |A|     fit     ext   total  n_obs    ARI    AMI
   dask      2  26250   0.621   0.316   0.937    264  1.000  1.000
   dask      4  13125   0.343   0.149   0.492    262  1.000  1.000
   dask      8   6563   0.187   0.153   0.340    257  1.000  1.000
   dask     16   3282   0.099   0.147   0.246    240  1.000  1.000
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
