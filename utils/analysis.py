####################################################################################################
# Introduction to Machine Learning
# Coursework 1 helper functions to read in and analyse training datasets
####################################################################################################

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def read_dataset(filepath):
    """
    Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file.

    Returns:
        tuple: returns a tuple of (X, y, classes), each being a numpy array.
            X is a numpy array with shape (N, K),
                where N is the number of instances
                K is the number of features/attributes
            y is a numpy array with shape (N, ), and each element should be
                an integer from 0 to C-1 where C is the number of classes
            classes : a numpy array with shape (C, ), which contains the
                unique class labels corresponding to the integers in y
    """

    # Create array of data
    data = np.genfromtxt(filepath, delimiter=",", dtype=str, encoding=None)

    # Slice x and y data
    X = data[:, :-1].astype(int)
    y = data[:, -1]
    y = np.array([str(label).strip() for label in y])

    classes = np.unique(y)

    return X, y, classes


# Print statistics for training data (question 1.1 and 1.2)
def analyze_dataset(x, y, classes, output_file):
    """
    Calculates the min, max, range, mean, median, standard deviation for an entire dataset and for
        each separate class saved at the filepath. These values are all printed
    Args:
        x (np.array): a numpy array with shape (N, K),
            where N is the number of instances
            K is the number of features/attributes
        y (np.array): a numpy array with shape (N, ), and each element should be
            an integer from 0 to C-1 where C is the number of classes.
        classes (np.array): a numpy array with shape (C, ), which contains the
            unique class labels corresponding to the integers in y
        output_file (str): a string containing the name of the file to append the results to
    """

    # Format column alignment for easier comparison between classes
    def format_stats(stats):
        return "  ".join(f"{val:6.3f}" for val in stats)

    with open(output_file, "a") as f:
        # Sense checks
        f.write("\n============================================================ Sense checks ============"
              "=================================================\n")
        f.write(f"\nShape of attributes: {x.shape}")
        f.write(f"\nShape of labels: {y.shape}")
        f.write(f"\nUnique classes: {classes}")

        # Statistics across full dataset
        f.write(
            "\n\n======================================================= Full dataset statistics ======"
            "=================================================\n")
        f.write("For each attribute across the whole dataset:\n")
        f.write(f"\nMin:    {format_stats(x.min(axis=0))}")
        f.write(f"\nMax:    {format_stats(x.max(axis=0))}")
        range_values = x.max(axis=0) - x.min(axis=0)
        f.write(f"\nRange:  {format_stats(range_values)}")
        f.write(f"\nMean:   {format_stats(x.mean(axis=0))}")
        f.write(f"\nMedian: {format_stats(np.median(x, axis=0))}")
        f.write(f"\nS.d.:   {format_stats(x.std(axis=0))}")

        # Frequency analysis
        f.write(
            "\n\n======================================================= Frequency of each class ======"
            "=================================================\n")
        unique, counts = np.unique(y, return_counts=True)
        total = y.size
        for class_label, class_count in zip(unique, counts):
            f.write(f"\nClass {class_label}: Frequency = {class_count}, "
                  f"Proportion = {class_count / total:.3f}")

        # Individual class analysis
        # Initialize dictionaries to store the statistics
        mins = {}
        maxs = {}
        ranges = {}
        means = {}
        medians = {}
        std_devs = {}

        # Iterate over each class
        for class_name, class_name in enumerate(classes):
            # Select the rows where the class label is the current class
            class_rows = x[y == class_name]

            # Compute and store the statistics for this class
            mins[class_name] = np.min(class_rows, axis=0)
            maxs[class_name] = np.max(class_rows, axis=0)
            ranges[class_name] = maxs[class_name] - mins[class_name]
            means[class_name] = np.mean(class_rows, axis=0)
            medians[class_name] = np.median(class_rows, axis=0)
            std_devs[class_name] = np.std(class_rows, axis=0)

        # Print the class results
        f.write(
            "\n\n===================================================== Individual class statistics ===="
            "=================================================\n")

        f.write("\nMin for each attribute in each class:")
        for class_name, min_val in mins.items():
            f.write(f"\nClass {class_name}: {format_stats(min_val)}")

        f.write("\n\nMax for each attribute in each class:")
        for class_name, max_val in maxs.items():
            f.write(f"\nClass {class_name}: {format_stats(max_val)}")

        f.write("\n\nRange for each attribute in each class:")
        for class_name, range in ranges.items():
            f.write(f"\nClass {class_name}: {format_stats(range)}")

        f.write("\n\nMeans for each attribute in each class:")
        for class_name, mean in means.items():
            f.write(f"\nClass {class_name}: {format_stats(mean)}")

        f.write("\n\nMedians for each attribute in each class:")
        for class_name, median in medians.items():
            f.write(f"\nClass {class_name}: {format_stats(median)}")

        f.write("\n\nStandard Deviations for each attribute in each class:")
        for class_name, std_dev in std_devs.items():
            f.write(f"\nClass {class_name}: {format_stats(std_dev)}")

def read_and_sort_datasets(file_path):
    """
    Reads a dataset in from a specified file and returns an array containing this data with the rows
        sorted in ascending order, starting with the left-most column.

    Args:
        file_path (str): The filepath to the dataset file.

    Returns:
        sorted_X (np.ndarray): A 2D numpy array of shape (N, K), where N is the number of instances,
            and K is the number of attributes/features. It is sorted in ascending order, starting
            with the left-most column.
        sorted_y (np.ndarray): A 1D numpy array of shape (N,), containing the class labels for each
            instance. It is sorted to match sorted_X.
        classes (np.ndarray): A 1D numpy array containing the unique class labels.
    """
    X, y, class_labels = read_dataset(file_path)

    sorted_indices = np.lexsort(X.T)
    sorted_X = X[sorted_indices]
    sorted_y = y[sorted_indices]

    return sorted_X, sorted_y, class_labels

def compare_datasets(y_true, y_pred, class_labels):
    """
    Compares the true labels (y_true) with the predicted labels (y_pred) for each instance in the
        dataset. Calculates the number of true positives (TP), false positives (FP), true negatives
        (TN), and false negatives (FN) for each class label specified in class_labels.

    Args:
        y_true (np.array): List of true class labels for each instance.
        y_pred (np.array): List of predicted class labels for each instance.
        class_labels (np.array): List of unique class labels present in the dataset.

    Returns:
        results (dict): Dictionary containing TP, FP, TN, FN counts for each class label.
        overall_proportion_mismatched (float): Overall proportion of misclassified instances in the dataset.
    """

    results = {label: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for label in class_labels}

    # Iterate over each instance in the dataset
    for i in range(len(y_true)):
        full_label = y_true[i]
        noisy_label = y_pred[i]

        # Iterate over each class label to update TP, FP, FN, TN accordingly
        for label in class_labels:
            if full_label == label:
                if noisy_label == label:
                    results[label]['TP'] += 1  # True Positive: Correctly identified label
                else:
                    results[label]['FN'] += 1  # False Negative: Missed identifying label
            else:
                if noisy_label == label:
                    results[label]['FP'] += 1  # False Positive: Incorrectly identified label
                else:
                    results[label]['TN'] += 1  # True Negative: Correctly rejected label

    # Calculate overall proportion mismatched
    total_instances = len(y_true)
    total_correct = sum(results[label]['TP'] for label in class_labels)
    overall_proportion_mismatched = 1 - (total_correct / total_instances)

    return results, overall_proportion_mismatched

def build_confusion_matrix(y_true, y_pred, class_labels):
    """
    Computes and returns a confusion matrix based on the results obtained from comparing true and
        predicted labels.

    Args:
        y_true (np.array): List of true class labels for each instance.
        y_pred (np.array): List of predicted class labels for each instance.
        class_labels (np.array): List of unique class labels present in the dataset.

    Returns:
        confusion_matrix (np.array): A confusion matrix where rows represent true labels and columns
            represent predicted labels.
    """
    num_classes = len(class_labels)
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_index[true]
        pred_idx = label_to_index[pred]
        confusion_matrix[true_idx, pred_idx] += 1

    return confusion_matrix

def calculate_metrics(results):
    """
    Computes and returns dictionaries relating to different model performance scores, with values
        for each class in the dataset, as well as scores relative to the dataset as a whole.

    Args:
        results (dict): Dictionary containing TP, FP, TN, FN counts for each class label.

    Returns:
        accuracy (dict): Accuracy scores for predictions for each class
        precision (dict): Precision scores for predictions for each class
        recall (dict): Recall scores for predictions for each class
        f1_score (dict): F1 scores for predictions for each class
        macro_scores (tuple): Tuple of macro precision, recall and F1 scores
        weighted_scores (tuple) Tuple of weighted precision, recall and F1 scores
    """
    class_labels = list(results.keys())
    num_classes = len(class_labels)

    # Initialize dictionaries to store metrics
    accuracy = {}
    precision = {}
    recall = {}
    f1_score = {}

    # Calculate metrics for each class
    for label in class_labels:
        TP = results[label]['TP']
        FP = results[label]['FP']
        FN = results[label]['FN']
        TN = results[label]['TN']

        precision[label] = TP / (TP + FP) if TP + FP > 0 else 0
        recall[label] = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label]) \
            if precision[label] + recall[label] > 0 else 0
        accuracy[label] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Calculate overall metrics (macro-average)
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))
    macro_f1_score = np.mean(list(f1_score.values()))

    # Weighted metrics calculation
    weighted_precision = sum(
        (precision[label] * (results[label]['TP'] + results[label]['FN'])) for label in class_labels) / sum(
        (results[label]['TP'] + results[label]['FN']) for label in class_labels)
    weighted_recall = sum(
        (recall[label] * (results[label]['TP'] + results[label]['FN'])) for label in class_labels) / sum(
        (results[label]['TP'] + results[label]['FN']) for label in class_labels)
    weighted_f1_score = sum(
        (f1_score[label] * (results[label]['TP'] + results[label]['FN'])) for label in class_labels) / sum(
        (results[label]['TP'] + results[label]['FN']) for label in class_labels)

    return accuracy, precision, recall, f1_score, (macro_precision, macro_recall, macro_f1_score), (weighted_precision, weighted_recall, weighted_f1_score)

def plot_heatmap(X, y, classes, stat_func=np.mean, title="Heatmap"):
    """
    Calculates a specified statistical measure for each attribute by class, plots a heatmap and
        saves it to the current directory.

    Parameters:
        X (np.ndarray): A 2D numpy array of shape (N, K), where N is the number of instances,
            and K is the number of attributes/features.
        y (np.ndarray): A 1D numpy array of shape (N,), containing the class labels for each instance.
        classes (np.ndarray): A 1D numpy array containing the unique class labels.
        stat_func (function): A numpy function to compute a statistical measure (default is np.mean).
            Other functions can be np.median, np.std, etc.
        title (str): The title for the heatmap.
    """
    # Initialize a matrix to hold the computed statistics for each class and attribute
    stats_matrix = np.zeros((len(classes), X.shape[1]))

    # Compute the statistic for each class and attribute
    for i, cls in enumerate(classes):
        cls_mask = y == cls  # Create a mask for instances of the current class
        stats_matrix[i, :] = stat_func(X[cls_mask], axis=0)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(stats_matrix, annot=True, fmt=".2f", cmap='viridis', cbar=True, linewidths=.5,
                yticklabels=classes)
    plt.title(title)
    plt.ylabel('Class')
    plt.xlabel('Attribute')

    # Save the heat map to analysis/part_1
    filename = title.replace(" ", "_").replace("/", "_") + ".png"
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, filename)
    plt.savefig(file_path)
    plt.close()

def attribute_range(x, axis=0):
    """
    Calculate the range (max - min) for each attribute.

    Args:
        x (np.ndarray): Input data.
        axis (int): The axis along which to compute the range. Default is 0.

    Returns:
        np.ndarray: The range of values for each attribute.
    """
    return np.max(x, axis=axis) - np.min(x, axis=axis)

def calculate_accuracy(y_true, y_pred):
    """
    Calculate the percentage accuracy of a set of predictions.

    Args:
        y_true (np.array): The true labels of the data. A 1D numpy array where each element
            corresponds to the true class label of an instance.
        y_pred (np.array): The predicted labels of the data. A 1D numpy array where each element
            corresponds to the predicted class label of an instance, as determined
            by the model.

    Returns:
        An accuracy score from 0 to 1 (0 being 100% false predictions and 1 100% true)
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def combine_predictions(models, X_test):
    """
    Combines predictions from multiple models using majority vote for each instance in the test set.
    Function assumes all models in the list are already trained and can make predictions.

    Args:
        models (list of model objects): A list of trained model objects that have a `predict` method.
            These models are used to predict the labels of the test set.
        X_test (np.ndarray): A 2D numpy array of the test set features. Each row represents an instance
            and each column represents a feature.

    Returns:
        combined (np.ndarray): A 1D numpy array of combined predictions for the test set instances.
            The prediction for each instance is the one that received the majority vote among all
            models. In case of a tie, the label that appears first when the labels are sorted is
            chosen.
    """
    # Gather predictions from each model
    predictions = np.array([model.predict(X_test) for model in models])

    # Determine the majority vote for each instance
    combined = np.empty(len(X_test), dtype=predictions.dtype)
    for i in range(len(X_test)):
        instance_predictions = predictions[:, i]
        unique_labels, counts = np.unique(instance_predictions, return_counts=True)
        combined[i] = unique_labels[counts.argmax()]

    return combined

