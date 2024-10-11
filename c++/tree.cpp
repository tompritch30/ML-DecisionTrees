#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <memory>

#include "tree.h"

using namespace std;

// NODE DEFINITIONS
bool Node::is_pure() {
    return set<string>(y.begin(), y.end()).size() == 1;
}

double Node::calculate_entropy(vector<string>& y) {
    map<string, int> counts;
    for (const string& label : y) {
        counts[label]++;
    }

    double entropy = 0.0;
    for (auto& pair : counts) {
        double p = pair.second / static_cast<double>(y.size());
        entropy -= p * log2(p);
    }
    return entropy;
}

double Node::calculate_information_gain(int attribute, double split_point) {
    vector<string> left_y, right_y;
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][attribute] <= split_point) {
            left_y.push_back(y[i]);
        } else {
            right_y.push_back(y[i]);
        }
    }

    if (left_y.empty() || right_y.empty()) return 0;

    double total_entropy = calculate_entropy(y);
    double left_entropy = calculate_entropy(left_y);
    double right_entropy = calculate_entropy(right_y);
    double left_weight = left_y.size() / static_cast<double>(y.size());
    double right_weight = right_y.size() / static_cast<double>(y.size());

    return total_entropy - (left_weight * left_entropy + right_weight * right_entropy);
}

void Node::find_split() {
    double best_gain = 0.0;
    for (size_t attribute = 0; attribute < X[0].size(); ++attribute) {
        set<double> unique_values;
        for (const auto& row : X) {
            unique_values.insert(row[attribute]);
        }

        for (double split_point : unique_values) {
            double gain = calculate_information_gain(attribute, split_point);
            if (gain > best_gain) {
                best_gain = gain;
                this->split_attribute = attribute;
                this->split_point = split_point;
            }
        }
    }
}

pair<pair<vector<vector<double>>, vector<string>>, pair<vector<vector<double>>, vector<string>>> Node::split_dataset() {
        pair<vector<vector<double>>, vector<string>> left;
        pair<vector<vector<double>>, vector<string>> right;

        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][split_attribute] <= split_point) {
                left.first.push_back(X[i]);
                left.second.push_back(y[i]);
            } else {
                right.first.push_back(X[i]);
                right.second.push_back(y[i]);
            }
        }
        return {left, right};
    }

string Node::most_common_class() const {
    map<string, int> label_counts;
    for (const string& label : y) {
        label_counts[label]++;
    }

    int max_count = 0;
    string most_common;
    for (const auto& [label, count] : label_counts) {
        if (count > max_count) {
            max_count = count;
            most_common = label;
        }
    }

    return most_common;
}

// TREE DEFINITIONS
void DecisionTreeClassifier::_fit_node(unique_ptr<Node>& node) {
    if (node->depth < max_depth && node->y.size() >= min_samples_split) {
        node->find_split();
        if (node->split_attribute != -1) {
            auto [left_subset, right_subset] = node->split_dataset();
            node->left = make_unique<Node>(left_subset.first, left_subset.second, node->depth + 1);
            node->right = make_unique<Node>(right_subset.first, right_subset.second, node->depth + 1);
            _fit_node(node->left);
            _fit_node(node->right);
        }
    }
}

void DecisionTreeClassifier::fit(vector<vector<double>> X, vector<string> y) {
    assert(X.size() == y.size());
    head = make_unique<Node>(X, y, 0);
    _fit_node(head);
    is_trained = true;
}

vector<string> DecisionTreeClassifier::predict(const vector<vector<double>>& X) {
    assert(is_trained);
    vector<string> predictions;
    for (const auto& x : X) {
        Node* node = head.get();
        while (node->split_attribute != -1) {
            if (x[node->split_attribute] <= node->split_point) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        predictions.push_back(node->most_common_class());
    }
    return predictions;
}
