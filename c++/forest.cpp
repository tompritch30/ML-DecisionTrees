#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <memory>
#include <thread>
#include <mutex>

#include "tree.h"
#include "forest.h"

using namespace std;

void RandomForestClassifier::fit(const vector<vector<double>>& X, const vector<string>& y) {
    trees.clear();
    int n_features = X[0].size();
    int n_samples = X.size();
    random_device rd;
    mt19937 g(rd());

    
    vector<std::thread> threads;
    std::mutex io_mutex;

    for (int i = 0; i < n_estimators; ++i) {
        threads.emplace_back([&, i] {            
            vector<vector<double>> X_sample;
            vector<string> y_sample;
            vector<int> features;

            // Bootstrap samples
            if (bootstrap) {
                uniform_int_distribution<> dis(0, n_samples - 1);
                for (int n = 0; n < n_samples; ++n) {
                    int idx = dis(g);
                    X_sample.push_back(X[idx]);
                    y_sample.push_back(y[idx]);
                }
            } else {
                X_sample = X;
                y_sample = y;
            }

            // Feature selection
            if (max_features < 0 || max_features > n_features) {
                features = select_features(n_features, n_features, g);
            }
            else {
                features = select_features(max_features, n_features, g);
            }

            // Train tree on selected features
            DecisionTreeClassifier tree(max_depth, min_samples_split);
            auto X_subsample = subset_features(X_sample, features);
            tree.fit(X_subsample, y_sample);

            // Lock for thread safety when accessing shared resources
            {
                std::lock_guard<std::mutex> guard(io_mutex);
                trees.push_back(make_pair(std::move(tree), features));
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

vector<string> RandomForestClassifier::predict(const vector<vector<double>>& X) {
    vector<vector<string>> all_predictions;
    for (auto& [tree, features] : trees) {
        auto X_subsample = subset_features(X, features);
        all_predictions.push_back(tree.predict(X_subsample));
    }

    size_t maxInnerSize = 0;
    for (const auto& innerVec : all_predictions) {
        maxInnerSize = std::max(maxInnerSize, innerVec.size());
    }

    // Majority vote
    vector<string> final_predictions(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        map<string, int> vote_count;
        for (auto& predictions : all_predictions) {
            vote_count[predictions[i]]++;
        }
        final_predictions[i] = majority_vote(vote_count);
    }

    return final_predictions;
}

vector<int> RandomForestClassifier::select_features(int n_select, int n_features, mt19937& g) {
    vector<int> features(n_features);
    iota(features.begin(), features.end(), 0);
    shuffle(features.begin(), features.end(), g);
    features.resize(n_select);
    return features;
}

vector<vector<double>> RandomForestClassifier::subset_features(const vector<vector<double>>& X, const vector<int>& features) {
    vector<vector<double>> X_subsample(X.size(), vector<double>(features.size()));
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < features.size(); ++j) {
            X_subsample[i][j] = X[i][features[j]];
        }
    }
    return X_subsample;
}

string RandomForestClassifier::majority_vote(const map<string, int>& vote_count) {
    pair<string, int> majority("", 0);
    for (auto& vote : vote_count) {
        if (vote.second > majority.second) {
            majority = vote;
        }
    }
    return majority.first;
}