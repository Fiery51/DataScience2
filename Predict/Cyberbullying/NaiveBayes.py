import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB  # Import Naive Bayes
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
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust number of features as needed
X = vectorizer.fit_transform(df["tweet_text"]).toarray()
y = df["cyberbullying_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the Naive Bayes classifier
# Use GaussianNB for continuous data like TF-IDF
# If you want to use MultinomialNB, consider using CountVectorizer instead of TF-IDF
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_smote, y_train_smote)

# Predictions and Evaluation
predictions = nb_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
