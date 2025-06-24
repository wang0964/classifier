
from src.features.build_features import create_dummy_vars
from src.data.make_dataset import load_and_preprocess_data
from src.models.predict_model import evaluate_model
from src.models.train_model import train_model
import warnings
warnings.filterwarnings("ignore")

# Set the path to the raw dataset
data_path = 'data/raw/heart_2020_cleaned.csv'

# Load and preprocess the raw dataset
df = load_and_preprocess_data(data_path)

# Perform one-hot encoding on categorical features and split into x (features) and y (target)
x, y = create_dummy_vars(df)

# Train the model and split out the test set for evaluation
model, x_test, y_test = train_model(x, y)

# Evaluate the trained model using the test set
recall, f1, precision, confusion_mat = evaluate_model(model, x_test, y_test)

# Print evaluation results
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'\nF1: {f1:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
