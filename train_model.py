"""
Train and evaluate an XGBoost classifier on concatenated node embeddings.

This module loads graph embeddings and drug-disease labels, performs data splitting,
cross-validation, hyperparameter search, and saves the final model.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)


def load_data(embedding_file, label_file):
    """
    Load the embeddings and label file from CSV.
    Assumes embeddings are in the 'topological_embedding' column and labels are in 'source' and 'target' columns.
    The 'id' column in the embeddings file will correspond to the drug and disease node ids in the labels file.
    """
    # Read node embeddings (id, topological_embedding)
    embeddings_df = pd.read_csv(embedding_file)

    # Read drug-disease label file (source, target, y)
    labels_df = pd.read_csv(label_file)

    # Map each node ID to its string embedding
    embeddings_dict = dict(
        zip(embeddings_df["id"], embeddings_df["topological_embedding"])
    )

    # Build feature matrix by concatenating source and target embeddings
    concatenated_embeddings = []
    labels = []

    # Iterate over the rows in the labels dataframe to fetch embeddings for both drug and disease
    for _, row in labels_df.iterrows():
        source_id = row["source"]  # Drug node id
        target_id = row["target"]  # Disease node id
        label = row["y"]  # Class label (0 or 1)

        # Get the embeddings for the drug (source) and disease (target) from the dictionary
        source_embedding_str = embeddings_dict.get(source_id, None)
        target_embedding_str = embeddings_dict.get(target_id, None)

        if source_embedding_str is not None and target_embedding_str is not None:
            # Convert embedding string back to numeric list
            source_embedding = [
                float(val) for val in source_embedding_str.strip("[]").split()
            ]
            target_embedding = [
                float(val) for val in target_embedding_str.strip("[]").split()
            ]

            # Concatenate the drug and disease embeddings to form a single feature vector (8 features total)
            concatenated_embedding = source_embedding + target_embedding
            concatenated_embeddings.append(concatenated_embedding)
            labels.append(label)

    # Convert the list of concatenated embeddings into a NumPy array
    X = np.array(concatenated_embeddings)
    y = np.array(labels)

    return X, y


def split_data(embeddings, labels, test_size=0.2, random_state=42):
    """
    Split embeddings and labels into train/test preserving class balance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return X_train, X_test, y_train, y_test


def cross_validate_model(X_train, y_train, model, n_splits=5):
    """
    Evaluate model performance via stratified K-fold cross-validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

    return cv_scores


def perform_random_search(X_train, y_train):
    """
    Search hyperparameter space for XGBoost using random sampling.
    """
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        "n_estimators": randint(50, 500),  # Number of trees
        "max_depth": randint(3, 15),  # Max depth of trees
        "learning_rate": uniform(0.01, 0.3),  # Learning rate
        "colsample_bytree": uniform(
            0.5, 0.5
        ),  # Fraction of features to consider for each tree
        "reg_alpha": uniform(0, 1),  # L1 regularization term
        "reg_lambda": uniform(0, 1),  # L2 regularization term
    }

    # Initialize the XGBoost model
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=5,  # Number of random combinations to try
        scoring="accuracy",  # Metric to optimise for
        cv=5,  # 5-fold cross-validation
        verbose=3,
        random_state=42,
        n_jobs=-1,
    )

    # Perform the random search
    random_search.fit(X_train, y_train)

    print("Best parameters found by RandomizedSearchCV:")
    print(random_search.best_params_)

    return random_search.best_estimator_


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Tune hyperparameters, train the final XGBoost model, and report test metrics.
    """
    print("Performing RandomizedSearchCV for hyperparameter tuning...")
    best_model = perform_random_search(X_train, y_train)

    # Fit the best model on the full training data
    best_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Evaluation: accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return best_model


def save_model(model, filename):
    """
    Serialize trained model to disk via joblib.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def train_ml_model(embedding_file, label_file, model_file):
    """
    Main function to load data, train model, and save the model.
    """
    # Step 1: Load data
    print("Loading data...")
    embeddings, labels = load_data(embedding_file, label_file)

    # Step 2: Split data into training and test sets (keep test set independent)
    X_train, X_test, y_train, y_test = split_data(
        embeddings, labels, test_size=0.2, random_state=42
    )

    # Step 3: Cross-validation on training data (without hyperparameter tuning)
    print("Performing cross-validation on training data...")
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    cross_validate_model(X_train, y_train, model)

    # Step 4: Train model with RandomizedSearchCV and evaluate on independent test set
    print("Training model with RandomizedSearchCV and evaluating on the test set...")
    trained_model = train_xgboost(X_train, y_train, X_test, y_test)

    # Step 5: Save the model
    save_model(trained_model, model_file)


def main():
    """
    Parse command-line arguments and invoke training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on graph embeddings"
    )

    # Input data files
    parser.add_argument(
        "--embedding_file", required=True, help="Path to the embeddings CSV file"
    )
    parser.add_argument(
        "--label_file", required=True, help="Path to the label CSV file"
    )

    # Output model file
    parser.add_argument(
        "--model_file", required=True, help="Path to save the trained model (.pkl)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the train_ml_model function
    train_ml_model(args.embedding_file, args.label_file, args.model_file)


if __name__ == "__main__":
    main()
