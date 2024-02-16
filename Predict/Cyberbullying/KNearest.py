import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(
    r"C:\Users\tyler\OneDrive\Desktop\GitHub Repositories\DataScience2\Predict\Cyberbullying\cyberbullying.csv"
)


# Function to clean text data
def clean_text(text):
    # Remove emojis
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^\w\s]", "", text)
    return text


def preprocess(df):
    print("Starting preprocessing...")

    # Clean the text data
    print("Cleaning text data...")
    df["tweet_text"] = df["tweet_text"].apply(clean_text)
    print("Text data cleaned.")

    # Create a CountVectorizer instance
    print("Creating CountVectorizer instance...")
    vectorizer = CountVectorizer(
        max_features=5000
    )  # Adjust the number of features as needed
    print("CountVectorizer instance created.")

    # Apply the vectorizer to the tweet texts to transform them into a Bag of Words model
    print("Applying CountVectorizer to tweet texts...")
    X = vectorizer.fit_transform(df["tweet_text"])
    print("CountVectorizer applied.")

    # Use 'cyberbullying_type' as the target variable
    print("Setting target variable...")
    y = df["cyberbullying_type"]
    print("Target variable set.")

    # Convert X to an array if it's not already
    print("Converting feature matrix to array...")
    X_array = X.toarray()
    print("Feature matrix converted to array.")

    # Split the data into training and testing sets (70% train, 30% test)
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y, test_size=0.3, random_state=42
    )
    print("Data split into training and testing sets.")

    # Reset the indexes and convert the Series to NumPy arrays for y_train and y_test
    print("Resetting indexes and converting to NumPy arrays...")
    y_train = y_train.reset_index(drop=True).to_numpy()
    y_test = y_test.reset_index(drop=True).to_numpy()
    print("Indexes reset and conversion to NumPy arrays completed.")

    print("Preprocessing completed.")
    return X_train, X_test, y_train, y_test


# Call preprocess and get the train/test splits
X_train, X_test, y_train, y_test = preprocess(df)


# Function to compute Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# KNN algorithm
class KNN:
    def __init__(self, k=3):
        print(f"Initializing KNN with k={k}")
        self.k = k

    def fit(self, X, y):
        print("Fitting model...")
        self.X_train = X
        self.y_train = y
        print("Model fitted.")

    def predict(self, X):
        print("Making predictions...")
        y_pred = [self._predict(x) for x in X]
        print("Predictions made.")
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances
        print("Computing distances...")
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        print("Distances computed.")

        # Sort and get k nearest neighbors
        print("Finding k nearest neighbors...")
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print("k nearest neighbors found.")

        # Most common label
        print("Determining the most common label...")
        most_common = Counter(k_nearest_labels).most_common(1)
        print("Most common label determined.")
        return most_common[0][0]
        # Function to find the best k


def find_best_k(X_train, X_test, y_train, y_test, min_k=1, max_k=20, step=2):
    best_k = min_k
    best_accuracy = 0

    print("Searching for the best k...")

    for k in range(min_k, max_k + 1, step):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"k = {k}: Accuracy = {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k


# Uncomment below lines to use the custom KNN algorithm
# knn_custom = KNN(k=3)
# knn_custom.fit(X_train, y_train)
# predictions_custom = knn_custom.predict(X_test)
# accuracy_custom = np.sum(predictions_custom == y_test) / len(y_test)
# print(f"Custom KNN Accuracy: {accuracy_custom}")

# Find the best k value
best_k = find_best_k(X_train, X_test, y_train, y_test)

# Using scikit-learn's KNeighborsClassifier with the best k
print(f"Using scikit-learn's KNeighborsClassifier with k = {best_k}...")
knn_sklearn = KNeighborsClassifier(n_neighbors=best_k)
knn_sklearn.fit(X_train, y_train)
predictions_sklearn = knn_sklearn.predict(X_test)

# Evaluate the model
accuracy_sklearn = np.mean(predictions_sklearn == y_test)
print(f"scikit-learn KNN with k = {best_k} Accuracy: {accuracy_sklearn}")

# Using scikit-learn's KNeighborsClassifier
print("Using scikit-learn's KNeighborsClassifier...")
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)
predictions_sklearn = knn_sklearn.predict(X_test)

# Evaluate the model
accuracy_sklearn = np.mean(predictions_sklearn == y_test)
print(f"scikit-learn KNN Accuracy: {accuracy_sklearn}")
