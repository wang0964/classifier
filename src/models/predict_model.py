# Import recall_score,f1_score,precision_score and confusion_matrix
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix as sk_confusion_matrix

# # Function to predict and evaluate


def evaluate_model(model, x_test, y_test):
    y_test_pred = model.predict(x_test)

    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    confusion_mat = sk_confusion_matrix(y_test, y_test_pred)

    return recall, f1, precision, confusion_mat
