import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'../..')))

from src.features.build_features import create_dummy_vars
from src.data.make_dataset import load_and_preprocess_data
from src.models.predict_model import evaluate_model
from src.models.train_model import train_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import pickle

# Set the path to the raw dataset
data_path = 'data/raw/heart_2020_cleaned.csv'

# Load and preprocess the raw dataset
df = load_and_preprocess_data(data_path)

# Perform one-hot encoding on categorical features and split into x (features) and y (target)
x, y = create_dummy_vars(df)

# Train the model and split out the test set for evaluation
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=8000, stratify=y, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=8000, stratify=y_train, random_state=42)

rf_pickle = open(r'models/model.pkl', 'rb')
model = pickle.load(rf_pickle)
rf_pickle.close()

y_test_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Validation Set)")
plt.show()
