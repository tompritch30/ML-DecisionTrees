import numpy as np
from numpy.random import default_rng
from classification import DecisionTreeClassifier
from classification_multiway import DecisionTreeClassifier as DecisionTreeClassifierMultiway

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """Generate n mutually exclusive splits at random."""
    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices

def train_test_k_fold(n_folds, X, y, random_generator=default_rng()):
    """Generate and train/test on k-fold splits."""
    n_instances = X.shape[0]
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    models = []
    scores = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.hstack([split_indices[i] for i in range(n_folds) if i != k])

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        model = DecisionTreeClassifier()
        #model = DecisionTreeClassifierMultiway()
        model.fit(X_train, y_train)  # Train model
        models.append(model)  # Store the trained model
        predictions = model.predict(X_test)  # Predict on test set
        accuracy = np.mean(predictions == y_test)  # Calculate accuracy
        print(accuracy)
        scores.append(accuracy)

    return models, scores

def cross_validation_report(X, y, n_folds=10):
    """Perform cross-validation and report metrics."""
    models, scores = train_test_k_fold(n_folds, X, y)
    print(f"Average accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return models
