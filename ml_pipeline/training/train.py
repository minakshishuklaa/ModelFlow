import joblib
import mlflow
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ml_pipeline.evaluation.evaluate import (
    evaluate_model,
    print_results,
    select_best_model
)
from ml_pipeline.preprocessing.preprocess import (
    load_data,
    prepare_data,
    create_preprocessor,
    split_data
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml_pipeline"
    / "models"
)

MODEL_PATH = MODEL_DIR / "best_model.pkl"

# --------------------------------------------------
# MLFLOW CONFIGURATION
# --------------------------------------------------

mlflow.set_experiment("ModelFlow-Customer-Churn")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data(DATA_PATH)


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

X, y = prepare_data(df)


# --------------------------------------------------
# CREATE PREPROCESSOR
# --------------------------------------------------

preprocessor = create_preprocessor(X)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = split_data(X, y)


# --------------------------------------------------
# MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------

trained_models = {}
results = {}

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training: {model_name}")
    print("=" * 50)

    # Start MLflow experiment
    with mlflow.start_run(run_name=model_name):

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        # Train
        pipeline.fit(X_train, y_train)

        trained_models[model_name] = pipeline

        print(f"{model_name} training completed.")

        # Evaluate
        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        results[model_name] = metrics

        # Print results
        print_results(
            model_name,
            metrics
        )

        # Log metrics to MLflow
        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"]
        )

        mlflow.log_metric(
            "precision",
            metrics["precision"]
        )

        mlflow.log_metric(
            "recall",
            metrics["recall"]
        )

        mlflow.log_metric(
            "f1_score",
            metrics["f1_score"]
        )

        # Log model name
        mlflow.set_tag(
            "model_name",
            model_name
        )


# --------------------------------------------------
# SELECT BEST MODEL
# --------------------------------------------------

best_model_name = select_best_model(results)

best_model = trained_models[best_model_name]

print("\n" + "#" * 60)
print("🏆 BEST MODEL")
print("#" * 60)

print(f"Model: {best_model_name}")
print(
    f"F1 Score: "
    f"{results[best_model_name]['f1_score']:.4f}"
)

print("#" * 60)

# --------------------------------------------------
# SAVE BEST MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved successfully!")
print(f"Model path: {MODEL_PATH}")