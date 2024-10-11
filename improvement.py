##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from ctypes import *

from classification import RandomForestClassifier

class Hyperparameters(Structure):
    _fields_ = [
        ("accuracy", c_double),
        ("n_estimators", c_int),
        ("tree_depth", c_int),
        ("min_samples_split", c_int),
        ("max_features", c_int),
    ]

def train_and_predict(X_train, y_train, X_test, X_val=None, y_val=None):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    
    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################

    # ====================================================================#
    #                  SNEAK 100: Optimisation done in C++!               #
    # ====================================================================#

    try:
        lib = CDLL("./c++/liboptimisation.so")

        # Flatten the datasets to pass to C++
        X_flat = X_train.ravel()
        y_flat = y_train.astype('S1').view('S1').ravel()

        X_flat_p = X_flat.ctypes.data_as(POINTER(c_double))
        y_flat_p = y_flat.ctypes.data_as(POINTER(c_char))

        lib.find_best_params.argtypes = [POINTER(c_double), c_size_t, c_size_t, POINTER(c_char), c_size_t, POINTER(Hyperparameters)]

        params = Hyperparameters()

        # Call the C++ function, passing the instance by reference
        lib.find_best_params(X_flat_p, X_train.shape[0], X_train.shape[1], y_flat_p, len(y_train), byref(params))

        # Access the hyperparameters in Python
        print(f"\nK-fold accuracy: {params.accuracy * 100}%")
        print(f"n_estimators: {params.n_estimators}")
        print(f"tree_depth: {params.tree_depth}")
        print(f"min_samples_split: {params.min_samples_split}")
        print(f"max_features: {params.max_features}")

    except Exception as e:
        print(f"An exception occurred: {e}")
        print("Falling back to reserve set of hyperparameters.")

        # Define your reserve hyperparameters
        reserve_hyperparameters = {
            "n_estimators": 30,
            "max_depth": None,
            "min_samples_split": 2,
            "max_features": 12,
            "bootstrap": True
        }

        # Use reserve hyperparameters
        params = Hyperparameters(reserve_hyperparameters["n_estimators"],
                                reserve_hyperparameters["max_depth"] if reserve_hyperparameters["max_depth"] is not None else -1,
                                reserve_hyperparameters["min_samples_split"],
                                reserve_hyperparameters["max_features"],
                                reserve_hyperparameters["bootstrap"])

    # Build the random forest with the found or reserve hyperparameters
    print("\nBuilding random forest with chosen parameters")
    forest = RandomForestClassifier(n_estimators=params.n_estimators,
                                    max_depth=None if params.tree_depth == -1 else params.tree_depth,
                                    min_samples_split=2,
                                    max_features=10000 if params.max_features == -1 else params.max_features,
                                    bootstrap=True)
    forest.fit(X_train, y_train)

    print(f"Predicting classifications from test dataset...")
    predicted = forest.predict(X_test)
    return predicted
        