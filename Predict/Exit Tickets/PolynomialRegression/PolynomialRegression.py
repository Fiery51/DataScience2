import numpy as np
import pandas as pd
import itertools

# Load the dataset
file_path = r"C:\Users\tyler\OneDrive\Desktop\GitHub Repositories\DataScience2\Predict\Exit Tickets\LinearRegression\cars.csv"
df = pd.read_csv(file_path)

# Extracting features and labels
X = df[["miles_driven", "age"]].values
y = df["sales_price"].values


# Function to create polynomial features
def create_polynomial_features(X, degree):
    n_samples, n_features = X.shape

    def iter_combinations():
        for total in range(1, degree + 1):
            for indices in itertools.product(range(n_features), repeat=total):
                yield (indices, total)

    combinations = list(iter_combinations())
    n_output_features = len(combinations)
    XP = np.empty((n_samples, n_output_features), dtype=X.dtype)
    for i, (indices, total) in enumerate(combinations):
        XP[:, i] = X[:, indices].prod(1)
    return XP


# Creating polynomial features
degree = 2
X_poly = create_polynomial_features(X, degree)

# Adding a column of ones for intercept
X_poly = np.concatenate((np.ones((X_poly.shape[0], 1)), X_poly), axis=1)

# Splitting the data into training and testing sets (70-30 split)
split_ratio = 0.7
split_index = int(len(X_poly) * split_ratio)
X_train, X_test = X_poly[:split_index], X_poly[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Linear regression using Pseudoinverse
def linear_regression(X, y):
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


# Train the model
theta = linear_regression(X_train, y_train)

# Print the coefficients
print("Coefficients:", theta)

feature_names = [
    "1",
    "miles_driven",
    "age",
    "miles_driven^2",
    "miles_driven*age",
    "age^2",
    "miles_driven*age^2",  # Additional term for the extra coefficient
]

equation = "sales_price = "
for i, coeff in enumerate(theta):
    if i > 0:
        equation += " + " if coeff >= 0 else " - "
        equation += f"{abs(coeff)}*{feature_names[i]}"
    else:
        equation += f"{coeff}"

print("Polynomial Regression Equation:")
print(equation)


# Prediction function
def predict(X, theta):
    return X.dot(theta)


# Predictions
y_train_pred = predict(X_train, theta)
y_test_pred = predict(X_test, theta)


# RMSE and R^2 functions
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# Evaluate the model
rmse_train = rmse(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
rmse_test = rmse(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Convert R2 score to percentage
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print(f"RMSE on Training Set: {rmse_train}")
print(f"R2 Percentage on Training Set: {r2_train_percentage}%")
print(f"RMSE on Test Set: {rmse_test}")
print(f"R2 Percentage on Test Set: {r2_test_percentage}%")
