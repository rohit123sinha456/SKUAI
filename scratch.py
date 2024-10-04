import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
# Load the dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)


os.environ["AWS_ACCESS_KEY_ID"] = "GMm4SwQs065M9iMaQ2cF"
os.environ["AWS_SECRET_ACCESS_KEY"] = "adzbR3Kg3eS0oROgBbBizbQp2F4XPhXWE6d0nsPL"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://localhost:9000"
mlflow.set_tracking_uri("http://localhost:5001")

# Start an MLflow run
with mlflow.start_run():

    # Log parameters
    n_estimators = 100
    max_depth = 3
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train the model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(clf, "random_forest_model")

    print(f"Model logged with accuracy: {accuracy}")
