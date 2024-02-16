import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


# Custom decision tree implementation with debug lines
def calculate_gini_index(groups, classes):
    print("Calculating Gini Index...")
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion * proportion
        gini += (1.0 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_best_split(dataset):
    print("Finding the Best Split...")
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = calculate_gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {"index": b_index, "value": b_value, "groups": b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    print(f"Splitting tree at depth: {depth}")
    left, right = node["groups"]
    del node["groups"]
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_best_split(left)
        split(node["left"], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_best_split(right)
        split(node["right"], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    print("Building decision tree...")
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


# [Remaining code for data loading and preprocessing stays the same]

# Convert the training set into a format suitable for the custom decision tree
train = np.column_stack((X_train_resampled, y_train_resampled))

# Building the custom decision tree
tree = build_tree(train.tolist(), max_depth=5, min_size=10)
print("Custom Decision Tree Model Built")

# Preparing test data for prediction
test = np.column_stack((X_test, y_test))

# Making predictions on the test set and evaluating
y_pred = [predict(tree, row) for row in test.tolist()]
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
