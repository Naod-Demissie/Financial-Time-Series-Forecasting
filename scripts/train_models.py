import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, Dropout, SimpleRNN
from tensorflow.keras.callbacks import ModelCheckpoint


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class ClassicalModelTrainer:
    """Trainer for classical machine learning models with MLflow tracking."""

    def __init__(self):
        """Initialize MLflow tracking URI and print initialization message."""
        self.mlflow_tracking_uri = "./mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        print("Initialized ModelTrainer with MLflow tracking.")

    def load_data(self, dataset_name):
        """Load train, validation, and test datasets from processed files."""
        print(f"Loading {dataset_name} dataset...")
        self.X_train = np.load(
            f"../data/processed/{dataset_name}_X_train.npy", allow_pickle=True
        )
        self.X_val = np.load(
            f"../data/processed/{dataset_name}_X_val.npy", allow_pickle=True
        )
        self.X_test = np.load(
            f"../data/processed/{dataset_name}_X_test.npy", allow_pickle=True
        )
        self.y_train = np.load(
            f"../data/processed/{dataset_name}_y_train.npy", allow_pickle=True
        )
        self.y_val = np.load(
            f"../data/processed/{dataset_name}_y_val.npy", allow_pickle=True
        )
        self.y_test = np.load(
            f"../data/processed/{dataset_name}_y_test.npy", allow_pickle=True
        )

        print(f"Shapes of loaded data for {dataset_name}:")
        print(f"X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
        print(f"X_val: {self.X_val.shape}, y_val: {self.y_val.shape}")
        print(f"X_test: {self.X_test.shape}, y_test: {self.y_test.shape}")

    def train_model(self, model_type, dataset_name):
        """Train a specified model type on the given dataset and save the best model."""
        self.load_data(dataset_name)
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }

        if model_type not in models:
            raise ValueError(
                "Invalid model type. Choose from Logistic Regression, Decision Tree, Random Forest, Gradient Boosting."
            )

        model = models[model_type]

        print(f"Training {model_type} on {dataset_name} dataset...")
        model.fit(self.X_train, self.y_train)

        model_path = (
            f"../checkpoints/best_{model_type.replace(' ', '_')}_{dataset_name}.joblib"
        )
        joblib.dump(model, model_path)
        print(f"Model saved as {model_path}")

        mlflow.sklearn.log_model(model, f"{model_type}_model_{dataset_name}")

    def evaluate_model(self, model_type, dataset_name):
        """Evaluate the trained model on validation and test sets with performance metrics."""
        model_path = (
            f"../checkpoints/best_{model_type.replace(' ', '_')}_{dataset_name}.joblib"
        )
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")

        for dataset, X, y in zip(
            ["Validation", "Test"], [self.X_val, self.X_test], [self.y_val, self.y_test]
        ):
            print(f"Evaluating on {dataset} set...")
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            print(f"Accuracy on {dataset} set: {accuracy:.4f}")
            cm = confusion_matrix(y, y_pred)
            self.plot_confusion_matrix(cm, model_type, dataset)
            print(classification_report(y, y_pred))

    def plot_confusion_matrix(self, cm, model_name, dataset):
        """Plot and display the confusion matrix for model evaluation."""
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-fraudulent", "Fraudulent"],
            yticklabels=["Non-fraudulent", "Fraudulent"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name} on {dataset} set")
        plt.show()


class DeepModelTrainer:
    """
    Trainer class for deep learning models, supporting CNN, RNN, and LSTM architectures.
    Handles data loading, model initialization, training, evaluation, and performance visualization.
    """

    def __init__(self):
        """Initialize the trainer and set up MLflow tracking."""
        self.mlflow_tracking_uri = "./mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        print("Initialized DeepModelTrainer with MLflow tracking.")

    def load_data(self, dataset_name):
        """Load dataset from preprocessed .npy files and store in class attributes."""
        print(f"Loading {dataset_name} dataset...")
        self.X_train = np.load(
            f"../data/processed/{dataset_name}_X_train.npy", allow_pickle=True
        )
        self.X_val = np.load(
            f"../data/processed/{dataset_name}_X_val.npy", allow_pickle=True
        )
        self.X_test = np.load(
            f"../data/processed/{dataset_name}_X_test.npy", allow_pickle=True
        )
        self.y_train = np.load(
            f"../data/processed/{dataset_name}_y_train.npy", allow_pickle=True
        )
        self.y_val = np.load(
            f"../data/processed/{dataset_name}_y_val.npy", allow_pickle=True
        )
        self.y_test = np.load(
            f"../data/processed/{dataset_name}_y_test.npy", allow_pickle=True
        )

        print(f"Shapes of loaded data for {dataset_name}:")
        print(f"X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
        print(f"X_val: {self.X_val.shape}, y_val: {self.y_val.shape}")
        print(f"X_test: {self.X_test.shape}, y_test: {self.y_test.shape}")

    def initialize_model(self, model_type, dataset_name):
        """Initialize a deep learning model of type CNN, RNN, or LSTM based on the dataset."""
        self.load_data(dataset_name)
        input_shape = (self.X_train.shape[1], 1)

        if model_type == "CNN":
            model = Sequential(
                [
                    Conv1D(
                        32, kernel_size=3, activation="relu", input_shape=input_shape
                    ),
                    Flatten(),
                    Dense(64, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )
        elif model_type == "RNN":
            model = Sequential(
                [
                    SimpleRNN(32, activation="relu", input_shape=input_shape),
                    Dense(64, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )
        elif model_type == "LSTM":
            model = Sequential(
                [
                    LSTM(32, activation="relu", input_shape=input_shape),
                    Dense(64, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )
        else:
            raise ValueError("Invalid model type. Choose from CNN, RNN, or LSTM.")

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train_model(self, model, model_type, dataset_name):
        """Train the specified deep learning model and save the best performing checkpoint."""
        self.X_train = self.X_train.reshape(-1, self.X_train.shape[1], 1)
        self.X_val = self.X_val.reshape(-1, self.X_val.shape[1], 1)

        checkpoint_path = f"../checkpoints/best_{model_type}_{dataset_name}.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        history = model.fit(
            self.X_train,
            self.y_train,
            epochs=6,
            validation_data=(self.X_val, self.y_val),
            callbacks=[checkpoint],
            verbose=1,
        )

        self.plot_training_progress(history, model_type, dataset_name)
        return model

    def evaluate_model(self, model, dataset_name):
        """Evaluate the trained model using the test dataset and display performance metrics."""
        self.X_test = self.X_test.reshape(-1, self.X_test.shape[1], 1)
        y_pred = model.predict(self.X_test)
        y_pred = y_pred > 0.5  # Converting probabilities to binary labels

        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(3.5, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non-fraudulent", "Fraudulent"],
            yticklabels=["Non-fraudulent", "Fraudulent"],
        )
        plt.title(f"Confusion Matrix for {dataset_name}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

        # Print classification report and accuracy score
        print(f"Classification Report for {dataset_name}:")
        print(classification_report(self.y_test, y_pred))

        print(
            f"Accuracy Score for {dataset_name}: {accuracy_score(self.y_test, y_pred)}"
        )

    def plot_training_progress(self, history, model_type, dataset_name):
        """Plot training accuracy and loss curves over epochs."""
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"{model_type} Accuracy on {dataset_name}")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{model_type} Loss on {dataset_name}")
        plt.legend()

        plt.show()
