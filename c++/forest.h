#ifndef FOREST_H
#define FOREST_H

#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <set>
#include <iostream>
#include <cmath>
#include <memory>

#include "tree.h"

using namespace std;

class RandomForestClassifier {
public:
    int n_estimators;
    int max_depth;
    int min_samples_split;
    int max_features;
    bool bootstrap;
    vector<pair<DecisionTreeClassifier, vector<int>>> trees;

    RandomForestClassifier(int n_estimators=100, int max_depth=-1, size_t min_samples_split=2,
                           int max_features=-1, bool bootstrap=true)
        : n_estimators(n_estimators), max_depth(max_depth), min_samples_split(min_samples_split),
          max_features(max_features), bootstrap(bootstrap) {}

    void fit(const vector<vector<double>>& X, const vector<string>& y);
    vector<string> predict(const vector<vector<double>>& X);

private:
    vector<int> select_features(int n_select, int n_features, mt19937& g);
    vector<vector<double>> subset_features(const vector<vector<double>>& X, const vector<int>& features);
    string majority_vote(const map<string, int>& vote_count);
};

#endif
