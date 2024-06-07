# FairlyUncertain Documentation

FairlyUncertain is a Python library for evaluating fairness and uncertainty in machine learning models. 

## Table of Contents
- [FairlyUncertain Documentation](#fairlyuncertain-documentation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Code Demonstration](#code-demonstration)
  - [Codebase Structure](#codebase-structure)
  - [How to Add a New Dataset](#how-to-add-a-new-dataset)
  - [How to Add a New Method](#how-to-add-a-new-method)
  - [License](#license)
  - [Citation](#citation)

## Installation

To install FairlyUncertain, you can use pip:

```bash
pip install fairlyuncertain
```

## Code Demonstration

We have created a Python notebook for demonstrating some of the core features of FairlyUncertain. You can find the notebook [here](https://colab.research.google.com/drive/1YJrT-EMe7DOan3jsusHLRpSl-r32Gvy9?usp=sharing) or in the `experiments` folder in the `demo.ipynb` file.

In addition, we have provided all of the code and results for the accompanying paper in the `experiments` folder.

## Codebase Structure

FairlyUncertain is structured as follows:

- `fairlyuncertain/`
  - `data/`: Contains the code for loading and caching datasets.
  - `methods/`: Contains the implementations of the methods.
  - `__init__.py`: Initializes the FairlyUncertain package.
  - `utils.py`: Contains utility functions used in the experiments.
  - `visualize.py`: Contains handy functions for visualizing results.
  - `benchmark.py`: Contains the benchmarking code for running experiments.

## How to Add a New Dataset

FairlyUncertain is built on modularity and extensibility. This allows for easy integration of new datasets and methods. Here are some guidelines on how to add a new dataset.

1. **Add Dataset Loader Function**

Implement a function to load your new dataset and save it as a new file with a descriptive name in `fairlyuncertain/data/`. The loader function must output a dictionary with the following keys: covariates `X`, targets `y`, group indicators `group`, and number of observations `n`. For examples, please refer to the existing dataset loaders in `fairlyuncertain/data/`.

Note: In order to reduce the effort of loading the dataset, please ensure that the dataset can be loaded with the loader function without any manual intervention.

2. **Update Data Loaders Dictionary**

Add the new dataset loader function to the dictionary of data loaders in `fairlyuncertain/data/__init__.py`. When the package is imported, the data loaders will be available for use.

Once the dataset it loaded once, it will be automatically cached for quick access in the future.

## How to Add a New Method

FairlyUncertain is built on modularity and extensibility. This allows for easy integration of new datasets and methods. Here are some guidelines on how to add a new method.

1. **Implement Method**

Implement the new method in a new file in `fairlyuncertain/methods/`. The file should contain a function that takes in an instance dictionary and, ideally, a `Model` class for training and evaluating. The function should output a dictionary with the predictions in the `pred` key and, if relevant, heteroscedastic uncertainty estimates in the `std` key. For examples, please refer to the existing method implementations in `fairlyuncertain/methods/`.

Note: For standardized evaluation, the training-testing split appears in the instance dictionary. The method should use the training data to train the model and the testing data to make predictions.

2. **Update Methods Dictionary**

Add the new method function to the dictionary of methods in `fairlyuncertain/methods/__init__.py`. When the package is imported, the method will be available for use. There are several lists of methods relevant to different tasks e.g., estimating uncertainty in regression or evaluating fairness in classification. If the new method fits into one of these categories, please add it to the relevant list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use FairlyUncertain in your research, please cite the following paper: Coming soon.