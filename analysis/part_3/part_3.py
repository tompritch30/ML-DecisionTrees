####################################################################################################
# Introduction to Machine Learning
# Coursework 1 analysis of the initial binary tree
####################################################################################################


import numpy as np
import os
from utils.analysis import (read_dataset, calculate_metrics, build_confusion_matrix, compare_datasets,
                            calculate_accuracy)
from classification import DecisionTreeClassifier
from utils.save import save_model_pkl


def main():
    # Train decision tree on three different files
    data_files = ["train_full", "train_sub", "train_noisy"]
    test_data = "test"

    X_test, y_test, classes = read_dataset(f"./data/{test_data}.txt")

    print(f"Training new models...\n")

    class_labels = ["A", "C", "E", "G", "O", "Q"]

    for data_file in data_files:

        print(f"Processing {data_file}...")
        X_train, y_train, classes = read_dataset(f"./data/{data_file}.txt")

        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)

        y_pred = tree.predict(X_test)

        print(f"\nWriting results for {data_file}...\n")
        results, overall_proportion_mismatched = compare_datasets(y_test, y_pred, class_labels)

        overall_accuracy = calculate_accuracy(y_test, y_pred)

        accuracy, precision, recall, f1_score, macro_metrics, weighted_metrics = calculate_metrics(results)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "part3_analysis_output.txt")

        # Append comparison results to the analysis text file
        with open(output_file, "a") as f:
            f.write(f"Analysis of performance of tree trained on {data_file}:\n")

            # Append overall metrics
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy: {overall_accuracy:.4f}\n")
            f.write(f"  Precision (Macro): {macro_metrics[0]:.4f}\n")
            f.write(f"  Recall (Macro): {macro_metrics[1]:.4f}\n")
            f.write(f"  F1 Score (Macro): {macro_metrics[2]:.4f}\n")
            f.write(f"  Precision (Weighted): {weighted_metrics[0]:.4f}\n")
            f.write(f"  Recall (Weighted): {weighted_metrics[1]:.4f}\n")
            f.write(f"  F1 Score (Weighted): {weighted_metrics[2]:.4f}\n\n")

            # Append confusion matrix
            confusion_matrix_result = build_confusion_matrix(y_test, y_pred, class_labels)
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(confusion_matrix_result, separator=', '))

            # Append class-wise metrics
            for class_label in class_labels:
                metrics = results[class_label]
                f.write(f"\n\nClass {class_label} Metrics:\n")
                f.write(f"  True Positives: {metrics['TP']}\n")
                f.write(f"  False Positives: {metrics['FP']}\n")
                f.write(f"  False Negatives: {metrics['FN']}\n")
                f.write(f"  Accuracy: {accuracy[class_label]:.4f}\n")
                f.write(f"  Precision: {precision[class_label]:.4f}\n")
                f.write(f"  Recall: {recall[class_label]:.4f}\n")
                f.write(f"  F1 Score: {f1_score[class_label]:.4f}\n")


if __name__ == "__main__":
    main()