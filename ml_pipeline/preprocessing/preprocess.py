# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer


# def load_data(file_path):
#     """Load the Telco Customer Churn dataset."""

#     df = pd.read_csv(file_path)

#     print(f"Dataset loaded successfully.")
#     print(f"Rows: {df.shape[0]}")
#     print(f"Columns: {df.shape[1]}")

#     return df


# def prepare_data(df):
#     """Clean the dataset and separate features from target."""

#     # Remove customer ID because it does not provide useful
#     # information for predicting churn.
#     df = df.drop(columns=["customerID"])

#     # TotalCharges contains some blank values and is stored as text.
#     df["TotalCharges"] = pd.to_numeric(
#         df["TotalCharges"],
#         errors="coerce"
#     )

#     # Remove rows where target is missing.
#     df = df.dropna(subset=["Churn"])

#     # Convert target into numerical values.
#     # No  -> 0
#     # Yes -> 1
#     df["Churn"] = df["Churn"].map({
#         "No": 0,
#         "Yes": 1
#     })

#     X = df.drop(columns=["Churn"])
#     y = df["Churn"]

#     return X, y


# def create_preprocessor(X):
#     """Automatically identify numerical and categorical columns."""

#     numerical_columns = X.select_dtypes(
#         include=["int64", "float64"]
#     ).columns.tolist()

#     categorical_columns = X.select_dtypes(
#         include=["object"]
#     ).columns.tolist()

#     print("\nNumerical columns:")
#     print(numerical_columns)

#     print("\nCategorical columns:")
#     print(categorical_columns)

#     # Numerical preprocessing
#     numerical_pipeline = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler())
#         ]
#     )

#     # Categorical preprocessing
#     categorical_pipeline = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             (
#                 "encoder",
#                 OneHotEncoder(
#                     handle_unknown="ignore"
#                 )
#             )
#         ]
#     )

#     # Combine both pipelines
#     preprocessor = ColumnTransformer(
#         transformers=[
#             (
#                 "numerical",
#                 numerical_pipeline,
#                 numerical_columns
#             ),
#             (
#                 "categorical",
#                 categorical_pipeline,
#                 categorical_columns
#             )
#         ]
#     )

#     return preprocessor


# def split_data(X, y):
#     """Split data into training and testing sets."""

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     print("\nData split completed.")
#     print(f"Training samples: {len(X_train)}")
#     print(f"Testing samples: {len(X_test)}")

#     return X_train, X_test, y_train, y_test

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_data(file_path):
    """Load the Telco Customer Churn dataset."""

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def prepare_data(df):
    """Clean the dataset and separate features from target."""

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()

    print(f"Duplicate rows removed: {duplicate_count}")

    # Remove unnecessary whitespace from column names
    df.columns = df.columns.str.strip()

    # Remove unnecessary whitespace from categorical/string values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()

    # Remove customer ID because it does not provide useful
    # information for predicting churn.
    df = df.drop(columns=["customerID"])

    # TotalCharges contains some blank values and is stored as text.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Replace infinite values with missing values
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # Remove rows where target is missing.
    df = df.dropna(subset=["Churn"])

    # Convert target into numerical values.
    # No  -> 0
    # Yes -> 1
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Check if any unexpected target values were found
    if df["Churn"].isna().sum() > 0:
        print(
            f"Warning: {df['Churn'].isna().sum()} "
            f"invalid target values found."
        )

        df = df.dropna(subset=["Churn"])

    # Display missing values before imputation
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) > 0:
        print("\nMissing values found:")
        print(missing_values)
    else:
        print("\nNo missing values found.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def create_preprocessor(X):
    """Automatically identify numerical and categorical columns."""

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


def split_data(X, y):
    """Split data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nData split completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test