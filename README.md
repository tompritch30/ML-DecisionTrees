# ML Decision Trees on the MNIST Dataset

## Overview

This project focuses on building and experimenting with decision tree classifiers for the MNIST dataset. The dataset contains images of handwritten digits (0-9). Your goal is to implement a decision tree to classify the digits accurately.

## Data

The `data/` directory contains the following datasets:

- `train_full.txt`: The full training dataset.
- `train_sub.txt`: A subset of the training dataset.
- `train_noisy.txt`: A noisy version of the training dataset.
- `validation.txt`: Dataset for model validation.
- `test.txt`: The official test dataset for final evaluation.

**Important:** Only use `test.txt` for final model evaluation. Use `validation.txt` for model optimization and tuning.

For debugging or testing implementation, use the smaller datasets provided:

- `toy.txt`
- `simple1.txt`
- `simple2.txt`

## Code

### `classification.py`

This script contains the class `DecisionTreeClassifier`. You need to implement the following methods:

- `train()`: Train the decision tree on the given training dataset.
- `predict()`: Use the trained decision tree to predict labels for unseen data.

### `improvement.py`

This script includes the `train_and_predict()` function. In this file, you will improve or enhance your decision tree classifier by adding optimizations or additional techniques.

### `example_main.py`

This is an example of how your code might be invoked during evaluation. It demonstrates the usage of the classes and functions defined in `classification.py` and `improvement.py`.

## Python and C++ Integration

### Data Preparation in Python

Prepare the dataset in Python, ensuring the features (X) and labels (y) are formatted appropriately. Typically, features are stored in NumPy arrays, and labels can be either lists or NumPy arrays of strings or integers.

### Data Serialization

Since C++ functions cannot directly handle Python types, serialize the data before passing it to C++. For multidimensional arrays, flatten them into one-dimensional arrays. Ensure strings are encoded, typically using UTF-8.

### Passing Data to C++

Use Python’s `ctypes` to interface with C++ functions. Compile the C++ code into a shared library (`.so` file for Unix, `.dll` for Windows). This shared library can then be accessed and called by Python.

### Hyperparameter Optimization in C++

On the C++ side, deserialize the data back into native structures (e.g., `std::vector` for arrays). Implement hyperparameter optimization to find the best parameters for your decision tree model. The optimization should be evaluated based on performance metrics like accuracy and F1 score.

### Retrieving Results

Once the optimization is complete, serialize the results (e.g., best hyperparameters and evaluation metrics) back into a format that can be passed to Python. The Python script should then deserialize the results for further use or analysis.

### Compilation Instructions

Before running the optimization, compile the C++ code into a shared library. Use the provided Makefile:

```bash
make
