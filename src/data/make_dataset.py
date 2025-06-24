import pandas as pd

# Function to load and preprocess heart disease dataset


def load_and_preprocess_data(data_path):
    # Read the actual dataset from the provided path
    df = pd.read_csv(data_path)

    # Convert 'HeartDisease' column: 'Yes' to 1, 'No' to 0
    df['HeartDisease'] = df['HeartDisease'].replace({'Yes': 1, 'No': 0})

    # Convert binary categorical columns to numerical (Yes/No to 1/0)
    df['Smoking'] = df['Smoking'].replace({'Yes': 1, 'No': 0})
    df['AlcoholDrinking'] = df['AlcoholDrinking'].replace({'Yes': 1, 'No': 0})
    df['Stroke'] = df['Stroke'].replace({'Yes': 1, 'No': 0})
    df['DiffWalking'] = df['DiffWalking'].replace({'Yes': 1, 'No': 0})

    # Convert 'Sex' column: 'Male' to 1, 'Female' to 0
    df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})

    # Convert more Yes/No columns to 1/0
    df['PhysicalActivity'] = df['PhysicalActivity'].replace(
        {'Yes': 1, 'No': 0})
    df['Asthma'] = df['Asthma'].replace({'Yes': 1, 'No': 0})
    df['KidneyDisease'] = df['KidneyDisease'].replace({'Yes': 1, 'No': 0})
    df['SkinCancer'] = df['SkinCancer'].replace({'Yes': 1, 'No': 0})

    # Return the cleaned and encoded DataFrame
    return df
