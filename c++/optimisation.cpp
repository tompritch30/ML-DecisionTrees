#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>
#include <thread>
#include <mutex>

#include "tree.h"
#include "forest.h"
#include "optimisation.h"

using namespace std;

mutex mtx;

pair<vector<vector<double>>, vector<string>> read_dataset(const string& filepath) {
    ifstream file(filepath);
    vector<vector<double>> X;
    vector<string> y;

    if (!file.is_open()) {
        cerr << "Failed to open file: " << filepath << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> instance;
        string value;
        string label;

        // Read values into vector until the last comma
        while (getline(ss, value, ',')) {
            // Before attempting to convert, check if this is potentially the last value (label)
            if (ss.peek() != EOF) {
                instance.push_back(stod(value)); // Safely convert numeric values
            } else {
                label = value; // The last value is the label, not a numeric feature
            }
        }

        X.push_back(instance);
        y.push_back(label);
    }

    return make_pair(X, y);
}

double calculate_accuracy(const vector<string>& y_true, const vector<string>& y_pred) {
    int correct_predictions = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) correct_predictions++;
    }
    return static_cast<double>(correct_predictions) / y_true.size();
}

double train_and_evaluate_fold(const vector<vector<double>>& X_train, const vector<string>& y_train, 
                               const vector<vector<double>>& X_test, const vector<string>& y_test,
                               int n_estimators, int max_depth, size_t min_samples_split,
                               int max_features, bool bootstrap) {
    RandomForestClassifier forest(n_estimators, max_depth, min_samples_split, max_features, bootstrap);
    forest.fit(X_train, y_train);
    vector<string> yp = forest.predict(X_test);
    return calculate_accuracy(y_test, yp);
}

double k_fold_cross_validation_multithreaded(const vector<vector<double>>& X, const vector<string>& y, 
                                             int k, int n_estimators, int tree_depth, size_t min_samples_split,
                                             int max_features, bool bootstrap) {
    int n = X.size();
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    vector<double> accuracies;
    vector<thread> threads;

    mutex acc_mtx;

    for (int i = 0; i < k; ++i) {
        int fold_size = n / k;
        int start_index = i * fold_size;
        int end_index = (i == k - 1) ? n : (i + 1) * fold_size;

        auto X_train_fold = make_shared<vector<vector<double>>>();
        auto y_train_fold = make_shared<vector<string>>();
        auto X_test_fold = make_shared<vector<vector<double>>>();
        auto y_test_fold = make_shared<vector<string>>();

        for (int j = 0; j < n; ++j) {
            if (j >= start_index && j < end_index) {
                X_test_fold->push_back(X[indices[j]]);
                y_test_fold->push_back(y[indices[j]]);
            } else {
                X_train_fold->push_back(X[indices[j]]);
                y_train_fold->push_back(y[indices[j]]);
            }
        }

        threads.emplace_back([=, &accuracies, &acc_mtx]() {
            double accuracy = train_and_evaluate_fold(*X_train_fold, *y_train_fold, 
                                                      *X_test_fold, *y_test_fold, 
                                                      n_estimators,
                                                      tree_depth,
                                                      min_samples_split,
                                                      max_features,
                                                      bootstrap);
            {
                lock_guard<mutex> acc_guard(acc_mtx);
                accuracies.push_back(accuracy);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    double total_accuracy = accumulate(accuracies.begin(), accuracies.end(), 0.0);
    double average_accuracy = total_accuracy / k;    

    return average_accuracy;
}

tuple<double, int, int, size_t, int> find_best_params(const vector<vector<double>>& X, const vector<string>& y) {
    int k = 5;
    // vector<int> n_estimators_options = {2, 5, 10};
    // vector<int> tree_depth_options = {5, 10, 15, 20, 25};
    // vector<size_t> min_samples_split_options = {2};
    // vector<int> max_features_options = {16};
    // bool bootstrap = true;

    vector<int> n_estimators_options = {5, 10, 25};
    vector<int> tree_depth_options = {12, 16, 20};
    size_t min_samples_split = 2;
    vector<int> max_features_options = {10, 12};
    bool bootstrap = true;

    double best_accuracy = 0.0;
    tuple<double, int, int, size_t, int> best_params;

    // ofstream file("hyperparameters_results_beech14.csv");
    // file << "n_estimators,tree_depth,min_samples_split,max_features,Accuracy\n";

    // Iterate through hyperparameter combinations
    for (int n_estimators : n_estimators_options) {
        for (int tree_depth : tree_depth_options) {
            for (int max_features : max_features_options) {
                // Call k-fold cross-validation
                cout << "n_estimators, tree_depth, min_samples_split, max_features => ";
                cout <<  "{ " << n_estimators << ", " << tree_depth << ", " << min_samples_split << ", " << max_features << " }" << endl;
                double average_accuracy = k_fold_cross_validation_multithreaded(X, y, k, n_estimators, tree_depth, min_samples_split, max_features, bootstrap);
                cout << "Accuracy: " << average_accuracy * 100 << "%" << endl;
                // file << n_estimators << "," << tree_depth << "," << min_samples_split << "," << max_features << "," << average_accuracy << "\n";

                if (average_accuracy > best_accuracy) {
                    best_accuracy = average_accuracy;
                    best_params = make_tuple(best_accuracy, n_estimators, tree_depth, min_samples_split, max_features);
                }
            }
        }
    }

    // file.close();
    return best_params;
}

int main() {
    string data = "train_full";    
    auto [X, y] = read_dataset("../data/" + data + ".txt");

    tuple<double, int, int, size_t, int> bestParams = find_best_params(X, y);
    
    cout << "Best Accuracy: " << get<0>(bestParams) * 100 << "%" << endl;
    cout << "Best Parameters: n_estimators=" << get<1>(bestParams) 
        << ", tree_depth=" << get<2>(bestParams) 
        << ", min_samples_split=" << get<3>(bestParams)
        << ", max_features=" << get<4>(bestParams) << endl;
    return 0;
}

extern "C" void find_best_params(double* X_data, size_t X_rows, size_t X_cols, char* y_data, size_t y_size, Hyperparameters* params) {
    // Reconstruct X
    vector<vector<double>> X(X_rows, vector<double>(X_cols));
    for (size_t i = 0; i < X_rows; ++i) {
        for (size_t j = 0; j < X_cols; ++j) {
            X[i][j] = X_data[i * X_cols + j];
        }
    }

    // Reconstruct y
    vector<string> y(y_size);
    for (size_t i = 0; i < y_size; ++i) {
        y[i] = string(1, y_data[i]);
    }

    tuple<double, int, int, size_t, int> bestParams = find_best_params(X, y);
    
    params->accuracy = get<0>(bestParams);
    params->n_estimators = get<1>(bestParams);
    params->tree_depth = get<2>(bestParams);
    params->min_samples_split = get<3>(bestParams);
    params->max_features = get<4>(bestParams);
}
