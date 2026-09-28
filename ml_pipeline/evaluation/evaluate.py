from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using multiple classification metrics.
    """

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def print_results(model_name, metrics):
    """
    Display model evaluation results.
    """

    print("\n" + "=" * 50)
    print(f"MODEL: {model_name}")
    print("=" * 50)

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['f1_score']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def select_best_model(results):
    """
    Select the model with the highest F1 score.
    """

    best_model_name = max(
        results,
        key=lambda model: results[model]["f1_score"]
    )

    return best_model_name