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
   
## Computational Environment
This project was benchmarked on a AWS HPC Cluster.  

- Scheduler: Slurm
- Head Node: t3.medium (Ubuntu 22.04)
- Compute Node: c6i.4xlarge (16 vCPUs / 32 GiB RAM)
- Region: eu-north-1 (Stockholm)

(details in `config.yaml`)  

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
- Number of ground-truth clusters: 5, 10
- Splits: 2, 4, 8, 16
- Dataset sizes: 5000, 200,000 samples
- Data Dimensionality: 10, 50
- Standard deviation of clusters: 1.0, 2.0
- Noise: 0, 0.05, 0.15
- Backends: seq, joblib, dask
- Number of CPU cores: 1, 2, 4, 8, 16

Each experiment is repeated using fixed random seeds to ensure reproducibility.

## Datasets
Experiments are run on two synthetic dataset variants:  

- blobs: clean Gaussian blobs generated with `make_blobs`.  
- noisy_blobs: the same blobs dataset + uniform noise points.  
  Noise points are sampled uniformly within the feature min/max range of the blob data.  
  These noise points are assigned the ground truth label '-1'.

## Project Structure
```
SDOclust-Parallel/
├── requirements.txt        # Required Python libraries (numpy, scikit-learn, joblib, dask)
├── config.yaml             # AWS ParallelCluster configuration (Region, Instances, Slurm)
├── parallel_kmeans.py      # Baseline implementations (KMeans, MiniBatchKMeans)
├── sdoclust_parallel.py    # Main pipeline: Parallel SDOclust + benchmarking suite
├── run.sbatch              # Slurm batch script for job submission
├── submit.sh               # Automation script to run experiments across core counts
│
├── plot_results.py         # Visualizes experimental result
├── logs/                   # Slurm standard output files (*.out)
│   └── sdoclust_bench_3.out 
├── results/                # Benchmark result data in CSV format
│   ├── results_core_1.csv  # Results with 1 core
│   ├── results_core_2.csv  # Results with 2 core
│   ├── results_core_4.csv  # Results with 4 core
│   ├── results_core_8.csv  # Results with 8 core
│   ├── results_core_16.csv # Results with 16 core
│   └── figures/            # Visualization plots
│
└── README.md               # Project documentation
```
Main script implementing the split-based SDOclust architecture and running all experiments.  
The project is organized to ensure reproducibility and clear logging:  

- sdoclust_parallel.py: The implementation of the parallel SDOclust and benchmarking.  
      Includes:  
      - fixed n-splits experiments  
      - sequential / joblib / dask comparisons  
      - baseline SDOclust comparison  
- logs/: Contains standard output (stdout). Files are named jobname_ID.out, allowing to trace the execution history.  
- results/: Contains raw performance data in CSV format (e.g., results_core_16.csv). These files store:  
      - Runtime (Fit, Extension, Total)  
      - Quality Metrics (ARI, AMI)  
      - Data Metadata (Noise level, dimensions, number of centers)  
      - Each row represents a unique combination of dataset, core count, and backends.  
      - If you execute code with `submit.sh`, each results are saving by number of cores.  
      - Visual plots generated from these CSVs can be found in the plots/ directory.  




## Execution
#### Local Execution
```
srun --cpu-bind=cores --cpus-per-task="16" \
      python sdoclust_parallel.py \
        --datasets "blobs,noisy_blobs" \
        --size "50000,200000" \
        --backends "seq,joblib,dask" \
        --splits "2,4,8,16" \
        --noise "0.05,0.15" \
        --centers "5,10" \
        --seeds "42" \
        --output "results/results.csv"
```
#### High-Performance Computing (Slurm)
```
chmod +x submit.sh
./submit.sh
```
submit.sh calls srun to be allocated the specified CPU cores (1–16) and automates the experiments.

**Arguments:**  

`--datasets`  Dataset types to evaluate.  
- blobs: clean Gaussian blobs  
- noisy_blobs: blobs with uniformly distributed noise points

`--size`  Number of samples: Multiple values can be provided as a comma separated list.

`--backends`  Label extension execution backend . 
- seq: sequential  
- joblib: multi thread    
- dask: task-based parallel execution

`--noise` Noise fractions (noisy_blobs).

`--center` Ground-truth cluster counts.
  
`--splits`  Number of data splits.

`--seeds`  Random seeds for reproducibility.

`--output` Path to result CSV file.

# Results

### calability by Core Count
- **Parallel Efficiency:** `parallel_sdoclust` demonstrates a clear downward trend in `total_time` as the number of CPU cores increases.    
- **Backend Comparison:** Both `joblib` and `dask` backends show superior scalability compared to the `seq` backend.

### Impact of Split Count
- Performance improves as the `n_splits` value increases, even when keeping the number of cores constant.
- In a 16 core environment, setting `n_splits=16` yielded the fastest total time, suggesting that data partitioning is reducing bottlenecks in parallel processing.

### Consistency in Clustering Accuracy
- **Accuracy:** Despite partitioning data and extending labels, there is insignificant difference in ARI and AMI scores compared to the baseline full model.
- **Dataset Performance:** Maintained performance of 1.0 on the `blobs` dataset and consistent high scores on the `noisy_blobs` dataset.

### Impact of Cluster Centers (centers)
- **Execution Time:** More clusters generate more observers, increasing the computational load during the label extension phase.
- **Accuracy:** SDOclust maintains robust ARI/AMI scores even as cluster complexity increases.
- **Parallel Benefit:** The speedup from using multiple cores is more significant at higher center counts, as the increased distance calculations are efficiently distributed.

### Benchmarking
- **Noise Robustness:** On the `noisy_blobs` dataset, the Parallel SDOclust achieved a higher ARI compared to `minibatch_kmeans`, proving superior accuracy in the presence of noise.
- **Speed vs Accuracy:** While `minibatch_kmeans` remains faster in absolute execution time, SDOclust significantly closes the gap through parallelization while providing higher classification accuracy.



### Result Notation
phase	method	dataset	N	d	centers	std	noise_frac	seed	backend	n_splits	splitA_pos	splitA_size	chunksize	knn_eff
- **phase**: identifier (baseline or experiment)
- **method**: executed algorithms (parallel_sdo, kmeans, minibatch etc.)
- **dataset**: used dataset (blobs, noisy_blobs)
- **N**: size of dataset
- **d**: data dimensionality
- **centers**: ground-truth cluster counts
- **std**: cluster standard deviation
- **noise_frac**: noise fractions
- **seed**: random seeds for reproducability
- **backend**: label extension method (seq, joblib or dask)
- **n_split**: number of splits (n)
- **splitA_pos**: split-A position (index of the specific split used to fit the observer model)
- **splitA_size**: number of samples in split-A
- **chunksize**: unit of data loaded into memory at once for distance calculations during the Label Extension phase
- **knn_eff**: value that determines how many observers each data point references to decide its cluster label during the Label Extension phase
- **fit_time**: time required to fit SDOclust on split-A   
- **ext_time**: time spent on label extension  
- **total_time**: sum of fitting and extension times  
- **n_obs**: Number of observers  
- **ARI**: Adjusted Rand Index   
- **AMI**: Adjusted Mutual Information
