import numpy as np
from classification import DecisionTreeClassifier, RandomForestClassifier
from utils.analysis import read_dataset, calculate_accuracy
from improvement import train_and_predict

if __name__ == "__main__":
    # Read dataset from local directory
    train = "train_full"
    test = "test"
    X_train, y_train, _ = read_dataset(f"./data/{train}.txt")
    X_test, y_test, _ = read_dataset(f"./data/{test}.txt")

    # ==========================================================

    # Train initial decision tree
    print(f"\nTraining decision tree on {train}.txt...")
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    print(f"Predicting classifications from test dataset...")
    yp = tree.predict(X_test)
    accuracy = calculate_accuracy(y_test, yp)
    print(f"Accuracy: {accuracy * 100}%")

    # ==========================================================

    # Train random forest with arbitrary hyperparameters
    print(f"\nTraining random forest on {train}.txt...")
    forest = RandomForestClassifier(n_estimators=25, 
                                  max_depth=None, 
                                  min_samples_split=2, 
                                  max_features=10, 
                                  bootstrap=True)
    forest.fit(X_train, y_train)

    print(f"Predicting classifications from test dataset...")
    yp = forest.predict(X_test)
    accuracy = calculate_accuracy(y_test, yp)
    print(f"Accuracy: {accuracy * 100}%")

    # ==========================================================

    # Train and optimise random forest using k-fold validation of the train dataset
    print(f"\nTraining and optimising random forest on {train}.txt...")
    yp = train_and_predict(X_train=X_train, y_train=y_train, X_test=X_test)
    accuracy = calculate_accuracy(y_test, yp)
    print(f"Accuracy: {accuracy * 100}%")
