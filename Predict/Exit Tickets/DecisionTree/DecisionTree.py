import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv(
    r"C:\Users\tyler\OneDrive\Desktop\GitHub Repositories\DataScience2\Predict\Exit Tickets\DecisionTree\penguins.csv"
)

df = df.dropna(subset=["Body Mass (g)"])
X = df.drop("Body Mass (g)", axis=1)
y = df["Body Mass (g)"]

# Split the data - 80% training, 20% testing
# Here, I'm omitting random_state for a different split each time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def encode_categorical_columns(df):
    """
    Encodes categorical text data into numbers for each column in the DataFrame.

    :param df: pandas DataFrame with categorical text data.
    :return: DataFrame with text data encoded as numbers.
    """
    encoded_df = df.copy()
    for column in encoded_df.select_dtypes(include=["object"]).columns:
        encoded_df[column] = encoded_df[column].astype("category").cat.codes

    return encoded_df


# Encode the DataFrame
encoded_df = encode_categorical_columns(df)
print(encoded_df.head())
