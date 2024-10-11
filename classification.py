#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from random import sample
from collections import Counter
# from graphviz import Digraph


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """


    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.head = None


    class Node(object):
        def __init__(self, X, y, depth):
            self.X = X
            self.y = y
            self.depth = depth
            self.split_attribute = None
            self.split_point = None
            self.left = None
            self.right = None


        # Check if all data in the node of the same classification
        def is_pure(self):
            return len(set(self.y)) == 1


        # Find the optimal attribute to split and the subsequent splitting value
        def find_split(self):
            best_gain = 0
            for attribute in range(len(self.X[0])):
                unique_values = sorted(set([x[attribute] for x in self.X]))
                for split_point in unique_values:
                    gain = self.calculate_information_gain(attribute, split_point)
                    if gain > best_gain:
                        best_gain = gain
                        self.split_attribute = attribute
                        self.split_point = split_point
            

        # Split the dataset based on the selected attribute and splitting value
        def split_dataset(self):
            left = {'X': [], 'y': []}
            right = {'X': [], 'y': []}
            for x, y in zip(self.X, self.y):
                if x[self.split_attribute] <= self.split_point:
                    left['X'].append(x)
                    left['y'].append(y)
                else:
                    right['X'].append(x)
                    right['y'].append(y)
            return left, right


        # Calculate the entropy of a given (sub)set of data
        def calculate_entropy(self, y):
            _, counts = np.unique(y, return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy if entropy != -np.inf else 0


        # Calculate the information gain given a splitting attribute and value
        def calculate_information_gain(self, attribute, split_point):
            left_y = [y for x, y in zip(self.X, self.y) if x[attribute] <= split_point]
            right_y = [y for x, y in zip(self.X, self.y) if x[attribute] > split_point]
            if not left_y or not right_y:
                return 0
            left_entropy = self.calculate_entropy(left_y)
            right_entropy = self.calculate_entropy(right_y)
            total_entropy = self.calculate_entropy(self.y)
            return total_entropy - (len(left_y) / len(self.y) * left_entropy + len(right_y) / len(self.y) * right_entropy)
        

    def fit(self, X, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        X (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert X.shape[0] == len(y), \
            "Training failed. X and Y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    

        # Generate the start of the decision tree and start building recurively
        self.head = self.Node(X, y, 0)
        self._fit_node(self.head)    
                    
        # Set a flag so that we know that the classifier has been trained
        self.is_trained = True
    

    def _fit_node(self, node):       
        if (self.max_depth is None or node.depth < self.max_depth) and len(node.y) >= self.min_samples_split:
            node.find_split()
            if node.split_attribute is not None:
                left_subset, right_subset = node.split_dataset()
                node.left = self.Node(left_subset['X'], left_subset['y'], node.depth + 1)
                node.right = self.Node(right_subset['X'], right_subset['y'], node.depth + 1)
                self._fit_node(node.left)
                self._fit_node(node.right)
    

    def print_tree(self, node=None, depth=0, indent="  "):
        if node is None:
            node = self.head

        if node.split_attribute is not None:
            print(f"{indent * depth}Node Depth {depth}: Split Attribute {node.split_attribute}, Split Point {node.split_point}")
            self.print_tree(node.left, depth + 1, indent)
            self.print_tree(node.right, depth + 1, indent)
        else:
            classification = max(set(node.y), key=node.y.count)  # Most common class
            print(f"{indent * depth}Leaf Depth {depth}: Classification {classification}")
       

    # def visualize_tree(self, node=None, graph=None, parent_name=None, depth=0):
    #     if node is None:
    #         node = self.head

    #     if graph is None:
    #         graph = Digraph()
    #         graph.attr(size="10,10")

    #     node_label = f"Depth {depth}\n"
    #     if node.split_attribute is not None:
    #         node_label += f"Split Attribute {node.split_attribute}\nSplit Point {node.split_point}"
    #     else:
    #         classification = max(set(node.y), key=node.y.count)
    #         node_label += f"Classification {classification}"

    #     node_name = f"node{depth}_{id(node)}"
    #     graph.node(name=node_name, label=node_label)

    #     if parent_name is not None:
    #         graph.edge(parent_name, node_name)

    #     if node.split_attribute is not None:
    #         self.visualize_tree(node.left, graph, node_name, depth + 1)
    #         self.visualize_tree(node.right, graph, node_name, depth + 1)

    #     if depth == 0:
    #         graph.attr(dpi='400')
    #         graph.render('./analysis/part2/decision_tree_visual', view=True, format='png')

    #     return graph

        
    def predict(self, X):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        X (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in X
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((X.shape[0],), dtype=object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
        predictions = [self._predict_instance(x, self.head) for x in X]
        return np.array(predictions)
    
    
    def _predict_instance(self, x, node):
        # Reached a leaf node
        if node.split_attribute is None:
            # Classify as the most common in the node
            return max(set(node.y), key=node.y.count)

        # Progress through the tree, left or right
        if x[node.split_attribute] <= node.split_point:
            return self._predict_instance(x, node.left)
        else:
            return self._predict_instance(x, node.right)


class RandomForestClassifier(object):
    """
    A simple random forest classifier.
    
    Attributes:
    n_estimators (int): The number of trees in the forest.
    max_depth (int): The maximum depth of the trees.
    min_samples_split (int): The minimum number of samples required to split an internal node.
    max_features (str, int, float): The number of features to consider when looking for the best split;
    bootstrap (bool): Whether bootstrap samples are used when building trees.
    
    Methods:
    fit(X, y): Builds a forest of trees from the training set (X, y).
    predict(X): Predicts the class for X.
    """
    

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features=10000, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.feature_indices = []


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        self.trees.clear()
        self.feature_indices.clear()

        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = np.random.choice(range(n_samples), n_samples)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y

            # Feature selection
            if isinstance(self.max_features, int) and self.max_features < n_features:
                n_select = self.max_features
                features = sample(range(n_features), n_select)
            else:
                features = list(range(n_features))            
            
            X_subsample = X_sample[:, features]

            # Train tree on selected features
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subsample, y_sample)
            self.trees.append(tree)
            self.feature_indices.append(features)


    def predict(self, X):
        X = np.array(X)
        all_predictions = []

        for tree, features in zip(self.trees, self.feature_indices):
            predictions = tree.predict(X[:, features])
            all_predictions.append(predictions)

        # Transpose to get predictions from each tree in rows
        all_predictions = np.array(all_predictions).T
        
        final_predictions = []
        for sample_predictions in all_predictions:
            count = Counter(sample_predictions)
            majority_vote = count.most_common(1)[0][0]
            final_predictions.append(majority_vote)

        return np.array(final_predictions)
    