import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


# Function to clean text data
def clean_text(text):
    # Remove emojis
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^\w\s]", "", text)
    return text


# Load the dataset
df = pd.read_csv(
    r"C:\Users\tyler\OneDrive\Desktop\GitHub Repositories\DataScience2\Predict\Cyberbullying\cyberbullying.csv"
)

# Clean the text data
df["tweet_text"] = df["tweet_text"].apply(clean_text)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=10000)  # Adjust number of features as needed
X = vectorizer.fit_transform(df["tweet_text"]).toarray()
y = df["cyberbullying_type"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data Split into Train and Test")

# Address class imbalance using SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest model
rf_model = RandomForestClassifier(
    max_depth=10, min_samples_leaf=10
)  # depth 5, minsamplesleaf = 10
rf_model.fit(X_train_resampled, y_train_resampled)
print("Random Forest Model Built")

# Make a prediction
print("Starting Prediction")
prediction = rf_model.predict(X_test[0].reshape(1, -1))
print(f"Prediction: {prediction[0]}")

# Making predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
