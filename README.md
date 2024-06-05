# FairlyUncertain Documentation

## Table of Contents
- [How to Add a New Dataset](#how-to-add-a-new-dataset)
- [How to Write a New Experiment](#how-to-write-a-new-experiment)
- [How to Add a New Model/Algorithm](#how-to-add-a-new-modelalgorithm)
- [How to Add a New Metric/Evaluation](#how-to-add-a-new-metriceval)

## How to Add a New Dataset

To add a new dataset, follow these steps.

1. **Import Necessary Libraries**

   Ensure you have the required libraries imported:
   ```python
   import pickle
   import os
   import numpy as np
   import pandas as pd
   import requests
   ```
3. **Create Dataset Loader Function**

Implement a function to load your new dataset. Here is an example:

  ```python
  def get_new_dataset():
      # Add logic to download, preprocess and return your dataset
      pass
  ```
3. **Update Data Loaders Dictionary**

Add the new dataset loader function to the dictionary of data loaders:

  ```python
  dataloaders = {
      'New Dataset': get_new_dataset,
      # other datasets...
  }
  ```
4. **Ensure Dataset Caching**

Make sure your dataset is cached for quick access:

  ```python
  def cache_dataset(name):
      # Logic to cache the dataset
      pass

  def read_dataset(name):
      # Logic to read the cached dataset
      pass
    ```
5. **Load Dataset Instance**

Ensure that the dataset can be loaded correctly:

  ```python
  def load_instance(name, train_split=.8):
      # Logic to load dataset instance
      pass
  ```

## How to Write a New Experiment
To write a new experiment, follow these steps:

1. **Import Necessary Libraries**

Ensure all required libraries are imported:

  ```python
  from tqdm import tqdm
  import numpy as np
  ...
  import fairlyuncertain as fu
  ```

2. **Define Algorithms to Test**

List the algorithms you want to test:

  ```python
  algorithm_names = ['Algorithm1', 'Algorithm2']
  algorithms = {name: fu.algorithms[name] for name in algorithm_names}
  ```

3. **Run Experiments**

Loop through datasets and algorithms to run your experiments:

  ```python
  results = {}
  for dataset in tqdm(yp.datasets):
      instance = fu.load_instance(dataset)
      results[dataset] = {'instance': instance}
      for algo_name in algorithms:
          results[dataset][algo_name] = algorithms[algo_name](instance)
  ```

4. **Evaluate and Plot Results**

Evaluate the results using the appropriate metrics and plot the outcomes:

  ```python
  for metric_name in fu.metrics:
      fu.plot_results(results, algorithms, fu.datasets, metric_name=metric_name)
  ```

## How to Add a New Model/Algorithm
To add a new model or algorithm, follow these steps:

1. **Add your Model Class if Necessary**

If your new algorithm requires a new model class, make those changes:

  ```python
  class Model:
      def __init__(self, **kwargs):
          # Initialization logic
          pass
  
      def fit(self, X, y):
          # Fit logic
          pass
  
      def predict(self, X):
          # Prediction logic
          pass
    ```

2. **Import Necessary Libraries**

Ensure necessary libraries and modules are imported:

  ```python
  import numpy as np
  ...
  import fairlyuncertain as fu
  ```

3. **Define the New Algorithm**

Implement the logic for your new algorithm:

  ```python
  def new_algorithm(instance, Model):
      # Algorithm implementation
      pass
  ```

4. **Add Algorithm to Dictionary**

Update the algorithms dictionary to include your new algorithm:

  ```python
  algorithms = {
      'New Algorithm': new_algorithm,
      # other algorithms...
  }
  ```

## How to Add a New Metric/Evaluation

To add a new metric or evaluation, follow these steps:

1. **Import Necessary Libraries**

Import required libraries for your metric:

  ```python
  import numpy as np
  ...
  ```

2. **Define the New Metric**

Implement the logic for the new metric:

  ```python
  def new_metric(pred, y, group):
      # Metric calculation logic
      pass
  ```

3. **Add Metric to Dictionary**

Update the metrics dictionary to include your new metric:

  ```python
  metrics = {
      'New Metric': new_metric,
      # other metrics...
  }
  ```

4. **Use New Metric in Evaluation**

Ensure the new metric is used in the evaluation process:

  ```python
  def evaluate_metrics(results, metrics):
      for metric_name in metrics:
          for dataset in results:
              results[dataset][metric_name] = metrics[metric_name](
                  results[dataset]['pred'],
                  results[dataset]['y'],
                  results[dataset]['group']
              )
  ```
