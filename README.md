# AIOps Data Splitting - Supplemental Materials
This repository contains the replication package for the paper "An Empirical Study of the Impact of Data Splitting Decisions on the Performance of AIOps Solutions".

## Introduction
We organize the replication package into four file folders.
1. Preprocessing: this folder contains code for extracting data from raw datasets and hyperparameter tuning;
2. Experiment: this folder contains code for our main experiment (i.e., evaluating concept drift detection methods and time-based ensemble approaches);
4. Results: this folder contains the results CSV files for our paper, including the results for metrics other than the AUC metric used in our paper;
3. Results analysis: this folder contains code for analyzing the dataset and experiment results.

Our code is based on the following packages and versions:

- Python: 3.8.3
- R: 3.6.3
- Numpy: 1.18.5
- Scipy: 1.5.0
- Pandas: 1.0.5
- Sklearn: 0.0
- Mkl: 2019.0

We recommend using an [Anaconda](https://docs.anaconda.com/anaconda/install/) environment with Python version 3.8.3, and every Python requirement should be met.

## Data preprocessing
This part contains code and materials for preprocessing the dataset. All code could be found under the `preprocessing` folder.

### Prepare dataset
We offer two approaches to prepare the data files: 1) download the data files provided by us on the [release page](https://github.com/SAILResearch/suppmaterial-19-yingzhe-aiops_data_splitting/releases); or 2) build the data files by yourself from the raw dataset and the preprocessing code provided by us.

#### Build the Backblaze data file
You could find the zipped CSV file (`disk_failure.zip`) for the Backblaze disk trace dataset. Unzip and place under the same folder as the experiment code files would work.
Otherwise, you could build the raw data file following the following steps:
1. Download the raw dataset from the [Backblaze website](https://www.backblaze.com/b2/hard-drive-test-data.html). 
   Our experiment used the disk stats data in 2015 (ZIP file for `2015 Data`).
2. Unzip all the raw data files and place all the CSV files (should be in the format `YYYY-MM-DD.csv`) in a single folder.
3. Update the `folder` variable at the head of the `preprocess_disk.py` file to store the CSV files and then execute the same Python file to extract samples. 
   Please be advised that it could take a while, and the output file would take several Gigabytes.

#### Build the Google data file
You could also download the prepared CSV file (`google_job_failure.zip`) from this project's release page for the Google cluster trace dataset.
Otherwise, you could build the same file following similar procedures to the Backblaze dataset using the `preprocess_google.py` file. You can find the raw cluster data [here](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md).

## Experiments
This part contains code for our main experiment in evaluating various model updating approaches. All code could be found under the `experiment` folder.

The experiment code accepts the following command-line arguments to select model, dataset, and iteration rounds for maximum flexibility.
1. `-d` is a **required** parameter for choosing the dataset. Two choices are available: `g` for the Google dataset, `b` for the Backblaze dataset.
2. `-m` is a **required** parameter for choosing the model. Five choices are available: `lr`, `cart`, `rf`, `gbdt`, and `nn`. Please note that the argument should be all *lowercase* letters.
3. `-n` is an optional parameter for the repetition time of the experiments. The default value is 100 iterations, which is also the same iteration number we used in our paper.

As the experiment could take a prolonged time to finish, we recommend executing them on a server with tools like `GNU Screen` or `nohup`. An example of evaluating the data splitting approaches on the `Google` data set and `RF` model in `100` iteration with `nohup` in the `background` and dump the command line output to `log.out` would be: `nohup python -u evaluate_data_splitting_approaches.py -d g -m rf -n 100 > log.out 2>&1 &`. 
Note that the following experiment code files rely on the `utilities.py` files in the same folder for helper functions.

We have the following experiment code available:
- `data_analysis.py` contains code for analyzing dataset-related statistics.
- `evaluate_data_leakage_challenge.py` contains code for proving the data leakage problem in AIOps datasets in our RQ1.
- `evaluate_data_splitting_approaches.py` contains code for evaluating random and time-based data splitting approaches in our RQ2.
- `evaluate_confusion_matrices.py` contains code for generating results for the confusion matrix analysis in the discussion part in our RQ2.
- `evaluate_model_update_approaches.py` contains code for evaluating the performance between static and periodically updated models in our RQ3.
- `evaluate_window_sizes.py` contains code for evaluating the window sizes in our RQ4.
- `evaluate_prequential_auc.py` contains code for generating results for the prequential AUC analysis in the discussion part in our RQ4.

## Experiment Results
This part contains the output data from our main experiments. All output CSV files could be found under the `results` folder.

## Results Analysis
This part contains code for the analysis of our datasets and experiment results. All code could be found under the `analysis` folder.
R script `result_analysis.R` contains code for plotting result figures in our paper using result files from the `results` folder.
