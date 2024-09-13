import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    return lr, params


def evaluate_model(lr, X_test, y_test):
    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average="macro")
    metrics['recall'] = recall_score(y_test, y_pred, average="macro")
    metrics['f1'] = f1_score(y_test, y_pred, average="macro")

    return metrics


def log_model(lr, X_train, metrics, params):
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Quickstart")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )

        return model_info


def model_inference(model_info, X_test, y_test):
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names

    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    return result



def main():
    # Traditional ML Code #
    # ------------------- #

    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Train model
    lr, params = train_model(X_train, y_train)

    # 3. Evaluate model
    metrics = evaluate_model(lr, X_test, y_test)

    # Logging with ML Flow #
    # --------------------- #

    # 4. Log model, parmas and metrics
    model_info = log_model(lr, X_train, metrics, params)

    # 5. Inference from a logged model
    result = model_inference(model_info, X_test, y_test)

    print(result.head())


if __name__ == '__main__':
    main()
