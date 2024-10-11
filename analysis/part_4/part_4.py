####################################################################################################
# Introduction to Machine Learning
# Coursework 1 analysis of the improved decision trees
####################################################################################################


from utils.analysis import read_dataset, calculate_metrics
from classification_multiway import DecisionTreeClassifier
from utils.save import save_model_jl


def main():

    # Train decision tree on three different files
    data_files = ["train_full", "train_sub", "train_noisy"]
    test_data = "test"

    # Read test dataset once, assuming it's the same across all trainings
    X_test, y_test, classes = read_dataset(f"./data/{test_data}.txt")

    print(f"Training new models using multiway tree...\n")

    for data_file in data_files:

        print(f"Processing {data_file}...")
        X_train, y_train, classes = read_dataset(f"./data/{data_file}.txt")

        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)
        # tree.visualize_tree()
        y_pred = tree.predict(X_test)
        save_model_jl(tree, f'./multi_way_{data_file}.joblib')

        calculate_metrics(y_test, y_pred)

if __name__ == "__main__":
    main()