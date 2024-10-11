####################################################################################################
# Introduction to Machine Learning
# Coursework 1 analysis of the provided training data
####################################################################################################


import numpy as np
import os
from utils.analysis import (read_dataset, analyze_dataset, compare_datasets, plot_heatmap,
                            calculate_metrics, build_confusion_matrix, read_and_sort_datasets,
                            calculate_accuracy)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    data_dir = os.path.join(project_root, 'data')

    datasets = {
        "train_full": os.path.join(data_dir, "train_full.txt"),
        "train_sub": os.path.join(data_dir, "train_sub.txt"),
        "train_noisy": os.path.join(data_dir, "train_noisy.txt")
    }

    output_file = os.path.join(script_dir, "part1_analysis_output.txt")

    class_labels = ["A", "C", "E", "G", "O", "Q"]

    # Load and process each dataset for analysis and visualization
    for dataset_name, filepath in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        X, y, classes = read_dataset(filepath)

        with open(output_file, "a") as f:
            f.write(f"\n\nAnalysing {filepath}:\n")
        analyze_dataset(X, y, class_labels, output_file)

        # Plot heatmap for the dataset
        title = f"Heatmap for {dataset_name} dataset"
        plot_heatmap(X, y, classes, np.mean, title)

    # Comparison analysis between full and noisy datasets
    print("\nComparing train_full and train_noisy datasets...")
    full_filepath = datasets["train_full"]
    noisy_filepath = datasets["train_noisy"]

    # Read and sort the datasets
    sorted_full_X, sorted_full_y, class_labels = read_and_sort_datasets(full_filepath)
    sorted_noisy_X, sorted_noisy_y, _ = read_and_sort_datasets(noisy_filepath)

    results, overall_proportion_mismatched = compare_datasets(sorted_full_y, sorted_noisy_y, class_labels)

    # Append comparison results to the analysis text file
    with open(output_file, "a") as f:
        f.write("Comparison between train_full and train_noisy datasets:\n")
        f.write(f"Overall Proportion of Mismatch: {overall_proportion_mismatched:.4f}\n\n")

        overall_accuracy = calculate_accuracy(sorted_full_y, sorted_noisy_y)

        # Calculate metrics
        accuracy, precision, recall, f1_score, macro_metrics, weighted_metrics = calculate_metrics(results)

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
        confusion_matrix_result = build_confusion_matrix(sorted_full_y, sorted_noisy_y, class_labels)
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