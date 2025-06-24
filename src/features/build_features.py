import pandas as pd

# Function to create dummy (one-hot encoded) variables from categorical columns


def create_dummy_vars(df):

    # Perform one-hot encoding on specified categorical columns
    # Each category will be transformed into a new binary column (0 or 1)
    df2 = pd.get_dummies(
        df, columns=['AgeCategory', 'Race', 'GenHealth', 'Diabetic'], dtype='int')

    # Save the processed DataFrame to a new CSV file (no index column)
    df2.to_csv('data/processed/Processed_heart_2020_cleaned.csv', index=None)

    # Separate the features (X) and target (y)
    # Drop the target column to get features
    x = df2.drop('HeartDisease', axis=1)
    y = df2['HeartDisease']               # Extract the target column

    return x, y  # Return features and target as separate variables
