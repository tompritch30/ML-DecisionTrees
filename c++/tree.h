#ifndef TREE_H
#define TREE_H

#include <memory>

using namespace std;

class Node {
public:
    vector<vector<double>> X;
    vector<string> y;
    int depth;
    int split_attribute = -1;
    double split_point = 0.0;
    unique_ptr<Node> left = nullptr;
    unique_ptr<Node> right = nullptr;

    Node(vector<vector<double>> X, vector<string> y, int depth) : X(X), y(y), depth(depth) {}
    Node(const Node& other)
        : X(other.X), y(other.y), depth(other.depth), split_attribute(other.split_attribute),
        split_point(other.split_point), left(nullptr), right(nullptr) {
        if (other.left != nullptr) {
            left = std::make_unique<Node>(*other.left); // Recursively copy left subtree
        }
        if (other.right != nullptr) {
            right = std::make_unique<Node>(*other.right); // Recursively copy right subtree
        }
    }

    bool is_pure();
    double calculate_entropy(vector<string>& y);
    double calculate_information_gain(int attribute, double split_point);
    void find_split();
    pair<pair<vector<vector<double>>, vector<string>>, pair<vector<vector<double>>, vector<string>>> split_dataset();
    string most_common_class() const;
};

class DecisionTreeClassifier {
private:
    int max_depth;
    size_t min_samples_split;
    unique_ptr<Node> head = nullptr;
    bool is_trained = false;

    void _fit_node(unique_ptr<Node>& node);

public:
    DecisionTreeClassifier(int max_depth = 15, int min_samples_split = 2) : max_depth(max_depth), min_samples_split(min_samples_split) {}
    DecisionTreeClassifier(const DecisionTreeClassifier& other)
        : max_depth(other.max_depth), min_samples_split(other.min_samples_split), is_trained(other.is_trained) {
        if (other.head != nullptr) {
            head = std::make_unique<Node>(*other.head);
        }
    }

    DecisionTreeClassifier& operator=(const DecisionTreeClassifier& other) {
        if (this != &other) { // Protect against self-assignment
            max_depth = other.max_depth;
            min_samples_split = other.min_samples_split;
            is_trained = other.is_trained;
            if (other.head != nullptr) {
                head = std::make_unique<Node>(*other.head); // Use Node's copy constructor
            } else {
                head = nullptr;
            }
        }
        return *this;
    }

    void fit(vector<vector<double>> X, vector<string> y);
    vector<string> predict(const vector<vector<double>>& X);
};


#endif
