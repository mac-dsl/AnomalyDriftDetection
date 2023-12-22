# Datasets and Dataset Generation

## Overview

This README documents the organization of files and methods used to generate the datasets for this repository.


## Directory Organization

The files in this repository that relevant for dataset generation have been listed below.
```.
├── README.md
├── data
│   ├── benchmark
│   │   ├── ECG
│   │   └── IOPS
│   └── synthetic
│       ├── p_drift
│       └── params_used.csv
├── moa_drift_generation.ipynb
├── requirements.txt
├── util
│   ├── combine_stream.py
│   ├── convert2arff.py
│   └── generate_moa_stream.py
└── view_drift.ipynb
```

## Requirements

The requirements for running the notebooks in this directory can be found in the [`requirements.txt`](../requirements.txt) file. These can be installed using pip or conda. The current working version of the code runs on Python 3.8.18.


## Datasets

The datasets for this repository can be found under the `data` directory. Datasets found under the `benchmark` subdirectory are published benchmark datasets for the anomaly detection and drift detection tasks. These files are used as the source files for generating drift streams. 

Datasets found under the `synthetic` subdirectory are datasets that have been generated for this paper. Further subdirectories contain data examples with variations of the named parameter. For example, the `p_drift` subdirectory contains different drift streams that vary in percentage of drift, with other parameters held to default values. The `p_drift/p5` subdirectory contains data examples corresponding to the group of streams categorized to be roughly 5% of drift. Drift stream file names also correspond to their characteristics in the following format: 

    `{dataset}_{drift type}_p{percentage drift}_n{number of drifts}_b{percentage before}.arff`

where dataset refers to the source benchmark dataset for the stream, drift type refers to gradual (grad) or abrupt (abr) and percentage drift, number of drifts, and percentage before are as described in the report.

**Note:** Current files may show with the naminng `a` instead of `b` to denote the percentage of drift which comes after anomalies. This will be refactored with changes to the code.


## Drift Stream Generation

The modules used to generate drift streams can be described by the following diagram:

![Architectural diagram of drift stream generation method](architecture_diagram.png)

The functions used to process ARFF files can be found in [`convert2arff.py`](../util/convert2arff.py). The functions used to algorithmically combine source streams and generate the corresponding MOA command can be found in [`combine_stream.py`](../util/combine_stream.py). The script for running the MOA commannds can be found in [`generate_moa_stream.py`](../util/generate_moa_stream.py).

The Jupyter Notebook [`moa_drift_generation.ipynb`](../moa_drift_generation.ipynb) shows how the functions can be used to create drift streams upon selecting parameter values. The notebook [`view_drift_generation.ipynb`](../view_drift.ipynb) shows how the plotting methods can be used to view generated streams.


## Author
- [Tammy Zeng](https://github.com/tammmyz)