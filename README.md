## Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.


### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class. Your task 
is to implement the ``train()`` and ``predict()`` methods.


- ``improvement.py``

	* Contains the skeleton code for the ``train_and_predict()`` function (Task 4.1).
Complete this function as an interface to your new/improved decision tree classifier.


- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods/functions defined in ``classification.py`` and ``improvement.py``.


### Instructions ragrding the link between Python and C+

#### Data Preparation in Python
The dataset, which consists of features X and labels y, is prepared in Python. The data should be in a format compatible with the C++ optimization function. For numerical data, numpy arrays are typically used, while labels can be a list or a numpy array of strings.

#### Serialization of Data
Since the C++ function cannot directly accept Python types such as numpy arrays or lists, the data must be serialized. This involves converting the data into a flat array if it's a multidimensional numpy array and encoding strings into a format such as UTF-8.

#### Passing Data to C++ Optimization Function
The serialized data is then passed to the C++ function. This is done using ctypes in Python, which allows calling C++ functions and passing C data types. The C++ function must be compiled into a shared library that ctypes can interact with.

#### Hyperparameter Optimization in C++
C++ receives the serialized data and reconstructs it into its native structures, such as std::vector for numerical arrays and std::string for labels. The optimization process iterates through different combinations of hyperparameters, training the machine learning model and evaluating its performance using metrics such as accuracy and F1 score.

#### Retrieval of Optimization Results
Once the optimization is complete, the results, which include the best-found hyperparameters and their corresponding evaluation metrics, are serialized back into a flat structure that can be passed back to Python.

#### Deserialization of Results in Python
Python receives the serialized results and deserializes them back into Python-readable formats, such as dictionaries or custom objects, for further analysis or use in the machine learning workflow.

#### Compilation Instructions
Before running the optimization, the shared library must be compiled from C++ source code. Use the provided Makefile for this purpose by running:

``make``

This will produce a liboptimisation.so file that the Python code can interface with.

#### Running the Optimization
The Python script that starts the process should be executed in an environment where the shared library is accessible. Ensure that the script correctly serializes the data, calls the optimization function, and handles the results.

For more details on the functions used, refer to the source files for Python and C++ code included in the project.

#### Dependencies
Python 3.x
Numpy
A C++ compiler supporting C++14 (e.g., g++)
C++ Standard Library

#### Notes
Ensure that the data passed to the C++ functions is correctly typed and matches the expected format of the C++ side.
For a new dataset or different machine learning model, the serialization and deserialization processes may need to be adjusted accordingly.
The optimization process is computationally intensive. Run it on a machine with adequate resources as per the specifications discussed previously.
By following these instructions, you should be able to perform hyperparameter optimization using a hybrid Python-C++ approach effectively.



