# AutoTest: 
This repo contains benchmarks, dataset, and code for **Auto-Test: Learning Semantic-Domain Constraints for Unsupervised Error Detection in Tables** (to appear in SIGMOD 2025). 

Our two manually-labeled benchmarks, **RT-bench** (with 1200 columns randomly sampled from real relational tables containing real errors) and **ST-bench** (with 1200 columns randomly sampled from spreadsheet tables containing real errors), can be found [here](https://github.com/qixuchen/AutoTest/tree/main/benchmarks)

Large data and model files (e.g., doduo models) used in the project that cannot be uploaded to GitHub can be found in this [google drive](https://drive.google.com/drive/folders/15pNNrk9IqXOMofR5b2cpftk4A-AfjQiN?usp=sharing).


## Overview:

Our "Auto-Test" approach automatically detects data errors in tables. It consists of two steps: 
- Step-1: At offline training time, using a large table corpora and in an unsupervised manner, it learns a new form of data-cleaning constraints that we call Semantic-Domain Constraints (SDCs);
- Step-2: At online prediction time, the learned SDCs can be applied efficiently at interactive speed (sub-second latency) to detect errors in new tables, without separate training required (because SDCs use a generic notion of semantic domains, like explained in the paper). 




## Use Auto-Test: 

There are two ways to use Auto-Test: 
- Only run Step-2 (easy):  we can reuse pre-trained SDCs from Step-1, to make new predictions on new tables
- Rerun both Step-1 and Step-2 (more time-consuming):  we can perform offline training from scratch to learn new SDCs, and then make predictions on new tables

We will describe both approaches below in turn.

### Only run Step-2 (easy): Reuse pre-trained SDCs to make new predictions

Since the offline SDC learning process can be time-consuming, we imagine that the most straightforward way to reuse Auto-Test is to directly apply the SDC constraints already learned offline, to new test tables where errors need to be detected. 

Please first refer to the [installation guide](#installation) to set up the depedencies and an environment named `VENV`. In the activated `VENV`, simply run

    python3 STEP3_SDC_application.py [PATH_TO_CSV_FILE] [PATH_TO_SDC_FILE]

where `[PATH_TO_CSV_FILE]` is the path to a CSV file on which errors need to be detected, and `[PATH_TO_SDC_FILE]` is path to a pre-trained SDC file learned offline.


#### A simple demo example

As an example to demonstrate, the following shows a comma-separated toy csv file in `./unit-test/example.csv` (similar to the demo example in our paper).

<pre>
    country,    statecode,  month,      city,       date
    Germany,    FL,         january,    mankanto,   12/3/2020
    Austria,    AZ,         <b>febuary</b>,    st peter,   11/5/2020
    France,     CA,         march,      seattle,    2/5/2021
    <b>Liechstein</b>, OK,         april,      saint paul, 10/23/2020
    Italy,      <b>germany</b>,    may,        shakopee,   10/7/2020
    Switzerland,AL,         june,       phoenix,    <b>new facility</b>
    Poland,     GA,         july,       <b>farimont</b>,   3/26/2021
</pre>

All Semantic-Domain Constraints (SDCs) learned offline are pre-populated in `./results/SDC`. There are three such SDC files, `RT-Train`, `ST-Train` and `Tablib`, corresponding to constraints learned from 3 different training corpora. We found that SDCs learnt on `RT-Train` perform the best, and will use this file located at `./results/SDC/rt_train_selected_sdc.csv`.


Run the following command:

    python3 ./online_detect.py ./unit_test/example.csv ./results/SDC/rt_train_selected_sdc.csv

After the script finishes, the detected outliers will be printed out. The following shows an example output.

    header      outlier         conf                val                                               
    date        new facility    0.991192            [10/7/2020, 11/5/2020, 3/26/2021, 10/23/2020, ...  
    month       febuary         0.987600            [july, may, april, june, febuary, january, march] 
    statecode   germany         0.977514            [germany, CA, OK, FL, AL, GA, AZ]                  
    country     Liechstein      0.962790            [Germany, Italy, Switzerland, Austria, Liechst...  

where `header` is the column header, `val` is the column values, `outlier` is the detected error, and `conf` is the confidence of the error.

### Rerun both Step-1 (time-consuming) and Step-2: Perform both offline learning and make new predictions 

If you want to rerun the entire AutoTest end-to-end, including the offline training in Step-1 (can be time-consuming) and online prediciton in Step-2, we prepared a script `AutoTest_end_to_end.sh` so that others can follow along and reproduce the results in the paper.

Before running the script, be sure to follow the [installation guide](#installation) and set everything correctly.

Then, run the script in `code/AutoTest` with the following command.

    source AutoTest_end_to_end.sh

**Note**: The offline training in Step-1 can be time consuming. On a Ubuntu 22.04 machine with 2.4GHz 64-core CPU and 512G memory, the training on `RT-train` took ~120 hours, and the training on `Tablib` took ~200 hours. The entire script (which reproduces the results on both corpora) can take ~320 hours in total.

After the script finishes, the resulting SDCs and pr-curves can be found in `code/AutoTest/results`:

1) The selected SDCs are in `code/AutoTest/results/SDC`.
2) The pr-curves are in `code/AutoTest/results/pr_curve`.

### Data: train and benchmark datasets

We make our training corpora (`RT-train`, `ST-train`) and the two 1200-column benchmark data  (`RT-bench`, `ST-bench`) available in `data` folder of our [google drive](https://drive.google.com/drive/folders/15pNNrk9IqXOMofR5b2cpftk4A-AfjQiN?usp=sharing) (described in Section 5.1 of the paper). 

#### Training corpora

Each line in the training corpora `RT-train` and `ST-train` corresponds to a extracted column. Each line has 4 fields (tab-separated) and the meaning of each field from left to right is as follows.


`fname`: the file name from where the column is extracted

`col_header`: header of the column

`dist_val_str` : string of distinct column values

`dist_val_count` : number of distinct column values

Values in `dist_val_str` are concatenated using `___`. For example, a column with `dist_val_str`

    a___b___c___d

corresponds to a column with 4 values a, b, c and d.

#### Two 1200-column benchmark data

Each row in `RT-bench` and `ST-bench` corresponds to a table column. 

Each row is described by the following fields:

`header`: header of the column

`ground_truth`: Set of **obvious** errors in this column (if any) 

`ground_truth_debateable`: Set of **contingent** errors in this column (if any)

`dist_val` : a list of distinct column values

`dist_val_count` : number of distinct column values


#### Synthetic corpus for SDC selection

The synthetic dataset used for SDC selection is stored in `synthetic.txt`. Each line corresponds to a synthesized column and has 4 fields (tab-separated).

`header`: header of the column

`ground_truth`: the synthesized error.

`dist_val` : a list of distinct column values

`dist_val_count` : number of distinct column values

#### Data cleaning benchmarks

The `experiments_using_data_cleaning_benchmarks` folder contains (1) the 9 input benchmark datasets used in our experiment in `data_cleaning_benchmark.txt` (Table 7 of our paper), and (2) the output from running our SDC in `data_clean_sdc.csv`, which shows the new SDC that can be applied on any columns in `data_cleaning_benchmark.txt`.




### Code: main repo

The code for the paper can be found under `code/AutoTest`. Before running the code, please follow the Installation section carefully for setup instructions.

#### (1) Offline SDC generation and quality assessment

The code for SDC generation and quality assessment can be found in `STEP1_SDC_generation.py`.

Usage: 

    python3 STEP1_SDC_generation.py [CORPUS_NAME]

where `[CORPUS_NAME]` is one of the values in `rt_train`, `st_train`, and `tablib`. 

This part takes an unlabeled large corpus as an input and mines SDC in a variety of domains (see the paper for details).
The mined SDCs are stored in `config.dir.project_base_dir/config.dir.project_base.sdc_output` by default, where `config.dir.project_base_dir` and `config.dir.project_base.sdc_output` are two directories set by you in `config.py` (see [configuration setup](#configuration-setup)).

The code for this part is expected to be slow on a personal device. 
It took us more than 100 hours to train on a corpus with ~200K columns, on a machine with 2.4GHz 64-core CPU and 512G memory. 

#### (2) Offline SDC selection

The code for coarse-grained and fine-grained SDC selection are in `STEP2_SDC_selection.py`.

Usage: 

    python3 STEP2_SDC_generation.py [CORPUS_NAME]

where `[CORPUS_NAME]` takes one of the values in `rt_train`, `st_train`, and `tablib`. 

The input is the set of SDCs mined from the previous step. A selected subset of SDCs that satisfies the specified constraints is returned.

As a demonstration, you may check our learned SDCs in `code/AutoTest/results/SDC` which contains SDCs selected by the fine-grained selection process, converted to human-readable format.

#### (3) Online inference using SDC

`STEP3_SDC_application.py` contains the code for applying mined rule on test columns to detect possible errors.

Usage:
    
    python3 STEP3_SDC_application.py [PATH_TO_BENCHMARK] [PATH_TO_SDC]

where `[PATH_TO_BENCHMARK]` is the path to one of the benchmark files (i.e., `RT-bench` or `ST-bench`), and `[PATH_TO_SDC]`is the path to the file storing the learned SDCs.

The result for this step, i.e., the detected errors, can be found in `code/AutoTest/results/detected_outliers`.

## Installation

The project is developed on a Ubuntu 20.04 system with 2.4GHz 64-core CPU, 512G memory, and Python version 3.7.16.

First, set up rust and cargo, which is required for SentenceBERT installation.

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Then restart your terminal and run

    source $HOME/.cargo/env

After downloading the repo, create a virtual environment `VENV`. Navigate into `code/AutoTest` and install all dependencies in the activated `VENV`:

    conda create -n VENV python=3.7
    conda activate VENV
    pip install --upgrade pip
    pip install tensorflow==1.15.5
    pip install -r requirements.txt

### Configuration setup

Before running the code, you need to specify the directories where you want store the results, datasets and SDCs in `config.py`. 
<!-- See the comments in `config.py` for the meaning of each directory.  -->

In general, there are two major directories that needs to be correctly set. The first one is the path specified by `config.dir.project_base_dir` which is the directory of the code base (i.e., where `code/AutoTest` is located).

The second one is the path specified by `config.dir.storage_root_dir` which is the directory where the corpora, benchmarks and intermediate results are stored.
Note that you need to reserve sufficiently large storage space for this directory. We recommend to reserve at least 200 - 300 GB, but more may be required if you want to try the code on larger training corpora. 

You may run `AutoTest_path_setup.py` and follow the instructions to check if everything is set up correctly.

    python3 ./AutoTest_path_setup.py

Remember to put (1) the training corpora, (2) the test benchmarks and (3) the synthetic dataset (for SDC selection) to the location as specified in `config.py`. Specifically,

1, Put RT_Train and ST_Train under `{config.dir.storage_root_dir}/{config.dir.storage_root.train_corpora}`.

2, Put RT_bench and ST_bench under `{config.dir.storage_root_dir}/{config.dir.storage_root.benchmark}`.

3, Put the synthetic dataset (for SDC selection) under `{config.dir.storage_root_dir}/{config.dir.storage_root.synth_dataset}`.

### Download GLoVe pretrained vectors

The GLoVe-related SDC requires GLoVe pretrained vectors. You may download it under under its [official website](https://nlp.stanford.edu/projects/glove/). The [6B version](https://nlp.stanford.edu/data/glove.6B.zip) is used in this project.

After downloading the zip file, unzip it and put the content under `{config.dir.storage_root_dir}/{config.dir.storage_root.glove}`.


### Load CTA models


We have separated the model files for Sherlock and Doduo in our [google drive](https://drive.google.com/drive/folders/15pNNrk9IqXOMofR5b2cpftk4A-AfjQiN?usp=sharing). You may find them under `code/sherlock-materials` and `code/doduo-materials`.

After downloading them, put them into the corresponding location under `AutoTest/sherlock-project` and `AutoTest/doduo-project` as follows.

For sherlock model:

```console
$ tree sherlock-project
sherlock-project
├── model_files
...
```

For doduo model and data:

```console
$ tree doduo-project
doduo-project
├── data
├── model
...
```


### Set up SentenceBERT and Sherlock  

SentenceBERT and Sherlock requires additional materials that need to be downloaded.

After the above configurations are correctly set, run the following script to set up SentenceBERT and Sherlock.

    python3 sbert_sherlock_setup.py



#### Minor note
It is noticed that the version of `multiprocessing` module used by this project may report the following bug when the training corpus is too large (~200K columns as what we used in the paper). 
You may see the following error message:

    struct.error: 'i' format requires -2147483648 <= number <= 2147483647

You can fix this by manually change several lines of code in the `multiprocessing` module, following [this post](https://github.com/open-mmlab/mmdetection/issues/2044). 

More specifically, the change is [this](https://github.com/python/cpython/commit/bccacd19fa7b56dcf2fbfab15992b6b94ab6666b).
