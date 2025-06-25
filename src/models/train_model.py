from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=8000, stratify=y, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=8000, stratify=y_train, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rfmodel = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    ).fit(x_train, y_train)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(rfmodel, f)

    return rfmodel, x_test, y_test
