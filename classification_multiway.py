#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed.
##############################################################################

import numpy as np
from itertools import combinations
from queue import Queue
from utils.data_loading import read_dataset
import math
import time 
import os


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """
    class LeafNode:
        def __init__(self, class_label=None):
            self.class_label = class_label
        
    class DecisionNode:
        def __init__(self, splitting_attribute=None, bin_value=None, labels=None):
            # The feature or attribute for the split condition
            self.splitting_attribute = splitting_attribute
            # Bin value for the split condition
            self.bin_value = bin_value
            # Child branches from the current node
            self.children = {}
            # Target labels 
            self.labels = labels
        
        def add_child(self, bin_value, child_node):
            self.children[bin_value] = child_node

        def get_child(self, attribute_value):
            # Find the corresponding bin for the attribute value 
            if attribute_value in self.children:
                return self.children[attribute_value]
            else:
                return 

    def __init__(self):
        self.is_trained = False
        self.root = None
        self.all_bin_combinations = Queue()
        self.current_bin_combination = ()

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(
            y
        ), "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        # Optimize the bin values by fitting the tree for different bin values 
        self.set_optimal_bins(x, y)

        discretized_data = self.discretize_dataset(data=x, use_new_bins=False)

        self.root = self.build_tree(
            data=discretized_data, target_labels=y, max_depth=30, depth=0
        )

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """

        # make sure that the classifier has been trained before predicting
        if self.is_trained is False:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # set up an empty (M, ) numpy array to store the predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # discretize the set
        discretized_data = self.discretize_dataset(data=x, use_new_bins=False)

        # Iterate through each instance in x and predict the label
        for i in range(discretized_data.shape[0]):
            instance = discretized_data[i, :]
            predicted_label = self.predict_instance(instance)
            predictions[i] = predicted_label

        return predictions

    def populate_bin_combinations(self):
        """Populates the queue with new combinations of bin boundaries."""

        # Define the initial bin boundaries
        bin_boundaries = np.arange(3, 17)

        # Generate new combinations with 5 bin boundaries
        new_combinations = [
            (list(sorted(combo)) + [math.inf])
            for combo in combinations(bin_boundaries, 5)
        ]

        # append infinity to allow for the last bin to be large enough to cover all remaining values
        new_combinations.append([math.inf])

        # Extend the queue with the new combinations
        self.all_bin_combinations.queue.extend(new_combinations)


    def discretize_dataset(self, data, use_new_bins=True):
        if self.all_bin_combinations.empty():
            self.populate_bin_combinations()

        if use_new_bins == True:
            self.current_bin_combination = self.all_bin_combinations.get()

        # print(self.current_bin_combination)

        discretized_data = np.zeros_like(data, dtype=int)

        for column in range(data.shape[1]):
            # Discretize the attribute values into custom bins
            discretized_data[:, column] = np.digitize(
                data[:, column], bins=self.current_bin_combination
            )

        return discretized_data

    def entropy(self, target_col):
        """
        This will determine the entropy of the data set.
        """
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = np.sum(
            [
                (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
                for i in range(len(elements))
            ]
        )
        return entropy

    def infoGain(self, target_labels, data):
        # Calculate the entropy for the whole dataset
        whole_entropy = self.entropy(target_labels)

        # Calculate values and their counts for split_attribute_val (bins)
        vals, counts = np.unique(data, return_counts=True)

        # Obtain weighted entropy for the given split_attribute (bins)
        weighted_entropy = np.sum(
            [
                (counts[i] / np.sum(counts))
                * self.entropy(target_labels[data == vals[i]])
                for i in range(len(vals))
            ]
        )

        # Calculate the information gain
        gain = whole_entropy - weighted_entropy
        return gain

    def bestSplitAttribute(self, data, target_labels, used_attributes):
        num_attributes = data.shape[1]

        # Exclude used attributes
        available_attributes = set(range(num_attributes)) - used_attributes

        # If no available attributes, return None
        if not available_attributes:
            return None

        # Find the best attribute among available attributes
        best_attribute = None
        current_max_gain = -1

        for attribute in available_attributes:
            attribute_data = data[:, attribute]

            gain = self.infoGain(target_labels, attribute_data)

            if gain > current_max_gain:
                current_max_gain = gain
                best_attribute = attribute

        return best_attribute

    def split_data(self, data, split_attribute, target_labels):
        feature_subsets = []
        label_subsets = []

        for bin_value in np.unique(data[:, split_attribute]):
            # Extract data for each bin value
            data_subset = data[data[:, split_attribute] == bin_value]
            labels_subset = target_labels[data[:, split_attribute] == bin_value]

            feature_subsets.append(data_subset)
            label_subsets.append(labels_subset)

        return feature_subsets, label_subsets

    def subset_pure(self, target_label):
        return np.all(target_label == target_label[0])

    def determine_class(self, target_labels):

        unique_classes, class_counts = np.unique(target_labels, return_counts=True)
        most_frequent_class = unique_classes[np.argmax(class_counts)]
        return most_frequent_class

    # Creating the tree

    def build_tree(self, data, target_labels, max_depth, depth=0, used_attributes=None):
        # Initialize used_attributes if not provided
        if used_attributes is None:
            used_attributes = set()

        # Stop criterion
        if depth == max_depth or self.subset_pure(target_labels):
            leaf_node = DecisionTreeClassifier.LeafNode(self.determine_class(target_labels))
            return leaf_node

        # Find the best split attribute excluding used attributes
        optimal_split_attribute = self.bestSplitAttribute(
            data, target_labels, used_attributes
        )

        if optimal_split_attribute is None:
            leaf_node = DecisionTreeClassifier.LeafNode(self.determine_class(target_labels))
            return leaf_node

        # Create node with splitting_attribute
        node = DecisionTreeClassifier.DecisionNode(
            optimal_split_attribute, bin_value=None, labels=target_labels
        )

        # Add the used attribute to the set
        used_attributes.add(optimal_split_attribute)

        # Split data
        feature_subsets, target_label_subsets = self.split_data(
            data, optimal_split_attribute, target_labels
        )

        for feature_subset, label_subset in zip(feature_subsets, target_label_subsets):
            # Check if feature_subset and label_subset are not empty
            if len(feature_subset) > 0 and len(label_subset) > 0:
                # Create the child node with the bin_value from the data
                bin_value = feature_subset[0, optimal_split_attribute]
                child_node = self.build_tree(
                    feature_subset,
                    label_subset,
                    max_depth,
                    depth + 1,
                    used_attributes.copy(),
                )
                node.add_child(bin_value, child_node)
        return node

    def predict_instance(self, instance, current_node=None, depth=0):
        # Initialize current_node to the root if not provided
        if depth == 0:
            current_node = self.root

        # Base case: leaf node
        if hasattr(current_node, "class_label"):
            return current_node.class_label

        # Obtain attribute value for given splitting attribute
        attribute_value = instance[current_node.splitting_attribute]

        # Retrieve the child node for the current attribute value
        child_node = current_node.get_child(attribute_value)

        # Check if child_node is None
        if child_node is None:
            # return the majority class at the given decision node,
            # since the instance attribute value does not correspond to bins at node
            return self.determine_class(current_node.labels)

        # Recursive call
        result = self.predict_instance(instance, child_node, depth + 1)
        return result

    def set_optimal_bins(self, train_data, train_labels):
        # used for a timer
        start_time = time.time()
        # max timer duration
        duration = 5 * 60
        bin_values_to_use = ()
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # load validation set
        validation_full_path = os.path.join(script_dir, "data/validation.txt")
        x_validation, y_validation, classes_validation = read_dataset(
            validation_full_path
        )
        max_accuracy = 0

        print(f"Checking for optimal bin values... This will take {duration / 60} minutes")

        # searches for optimal bin values that can be created within 5 minutes
        while (time.time() - start_time) < duration:
            # discretize Data
            discretized_data = self.discretize_dataset(data=train_data)

            # train classifier
            self.root = self.build_tree(
                data=discretized_data, target_labels=train_labels, max_depth=30, depth=0
            )

            # set a flag so that we know that the classifier has been trained
            self.is_trained = True

            # calculating predictions with the given bin values
            predictions = self.predict(x_validation)

            # this was done for debugging purposes
            # count_none = np.count_nonzero(predictions == None)
            result = np.equal(predictions, y_validation)
            accuracy = np.sum(result) / np.size(result)

            # accuracy being printed for debugging purposes
            # print(f"the current bin values are {current_bin)values}")
            # print(accuracy)

            # if the accuracy is greater than max_accuracy,
            # then save it as max_accuracy and store associated bins
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                bin_values_to_use = self.current_bin_combination

            # the ideal scenario is to stop the loop once all
            # combinations have been analyzed, would require extensive times
            if self.all_bin_combinations.empty():
                break

        self.current_bin_combination = bin_values_to_use
