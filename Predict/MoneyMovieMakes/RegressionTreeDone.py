import pandas as pd
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df, features):
    for feature in features:
        if df[feature].dtype == "object":
            df[feature] = df[feature].fillna(df[feature].mode()[0])
        else:
            df[feature] = df[feature].fillna(df[feature].median())
    return df


def get_feature_combinations(features, max_features):
    all_combinations = []
    for r in range(1, max_features + 1):
        combinations = itertools.combinations(features, r)
        all_combinations.extend(combinations)
    return all_combinations


def train_and_evaluate(X_train, y_train, X_test, y_test):
    parameters = {
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2", None],
    }
    dtree = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=dtree,
        param_grid=parameters,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, grid_search.best_params_


def evaluate_model(df, features, target):
    X = df[features]
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mse, best_params = train_and_evaluate(X_train, y_train, X_test, y_test)
    rmse = np.sqrt(mse)
    mean_target = y.mean()
    accuracy = 100 * (1 - rmse / mean_target)  # Negative accuracy is possible
    return rmse, accuracy, best_params


def main():
    filepath = r"C:\Users\tyler\OneDrive\Desktop\GitHub Repositories\DataScience2\Predict\MoneyMovieMakes\train.csv"
    df = load_data(filepath)
    target = "revenue"
    potential_features = [
        "budget",
        "popularity",
        "runtime",
        "original_language",
        "status",
    ]
    max_features = 10
    feature_combinations = get_feature_combinations(potential_features, max_features)

    best_rmse = float("inf")
    best_accuracy = float("-inf")
    best_features = None
    best_hyperparameters = None

    for features in feature_combinations:
        df_preprocessed = preprocess_data(df.copy(), list(features) + [target])
        rmse, accuracy, best_params = evaluate_model(
            df_preprocessed, list(features), target
        )
        print(
            f"Features: {features}, RMSE: {rmse}, Accuracy: {accuracy:.2f}%, Best Hyperparameters: {best_params}"
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_accuracy = accuracy
            best_features = features
            best_hyperparameters = best_params

    print(
        f"\nBest Model:\nFeatures: {best_features}, RMSE: {best_rmse}, Accuracy: {best_accuracy:.2f}%, Hyperparameters: {best_hyperparameters}"
    )


if __name__ == "__main__":
    main()
