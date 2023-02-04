"""
Library of functions for project 'Predict Customer Churn'

name: Nchey Mbamalu
data: Feb 2023
"""
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report


def import_data(path: str) -> pd.DataFrame:
    """
    Reads in path as a Pandas DataFrame

    Args:
        path: (str) path to the input data

    Returns:
        df_: (pd.DataFrame) Pandas DataFrame
    """
    df_ = pd.read_csv(path, index_col=0)
    target_encoder = {"Existing Customer": 0, "Attrited Customer": 1}
    df_ = (df_
           .assign(Attrition_Flag=df_["Attrition_Flag"].map(target_encoder))
           .rename({"Attrition_Flag": "Churn"}, axis=1)
           .drop("CLIENTNUM", axis=1)
           .copy(deep=True)
           )

    return df_


def perform_eda(dataframe: pd.DataFrame, path: str) -> None:
    """
    Performs EDA on dataframe and saves images to path

    Args:
        dataframe: (pd.DataFrame) Pandas DataFrame
        path: (str) folder where the image files
        will be saved to

    Returns:
        None
    """
    df_ = dataframe
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_.drop("Churn", axis=1).corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.tight_layout()
    plt.savefig(f"{path}/heatmap.png")

    cols = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct"
    ]

    for col in cols:
        plt.figure(figsize=(20, 10))
        if col in cols[:2]:
            df_[col].hist()
            plt.savefig(f"{path}/{col.lower()}_distribution.png")
        elif col == cols[2]:
            df_[col].value_counts(normalize=True).plot(kind="bar")
            plt.savefig(f"{path}/{col.lower()}_distribution.png")
        else:
            sns.histplot(df_[col], stat="density", kde=True)
            plt.savefig(f"{path}/total_transaction_distribution.png")


def encoder_helper(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that coverts each categorical feature to
    an encoded feature, with the average churn for each category

    Args:
        dataframe: (pd.DataFrame) Pandas DataFrame

    Returns:
        df_: (pd.DataFrame) Pandas DataFrame with encoded columns
    """
    df_ = dataframe.copy(deep=True)
    cat_cols = df_.drop("Churn", axis=1).select_dtypes("object").columns
    for col in cat_cols:
        categories = df_.groupby(col)["Churn"].mean().index.tolist()
        churn_averages = df_.groupby(col)["Churn"].mean().tolist()
        df_[f"{col}_Churn"] = df_[col].map(dict(zip(categories, churn_averages)))

    return df_


def perform_feature_engineering(df_: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits df_ into train and test sets

    Args:
        df_: (pd.DataFrame) Pandas DataFrame with encoded columns

    Returns:
        x_train (pd.DataFrame): Train set feature matrix
        x_test (pd.DataFrame): Test set feature matrix
        y_train (pd.Series): Train set target vector
        y_test (pd.Series): Test set target vector
    """
    features = df_.drop("Churn", axis=1).select_dtypes("number").columns.tolist()
    x_train, x_test, y_train, y_test = train_test_split(df_[features],
                                                        df_["Churn"],
                                                        test_size=0.3,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def train_models(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    Trains the models, generates their test set ROC
    curves, and saves the models and curves to their
    corresponding folders

    Args:
        x_train: (pd.DataFrame) Train set feature matrix
        x_test: (pd.DataFrame) Test set feature matrix
        y_train: (pd.Series) Train set target vector
        y_test: (pd.Series) Test set target vector

    Returns:
        None
    """
    paths = [
        "models/rfc_model.pkl",
        "models/logistic_model.pkl",
        "images/results/roc_curve_result.png"
    ]

    models = [
        RandomForestClassifier(random_state=42),
        LogisticRegression(solver="lbfgs", max_iter=3000)
    ]

    param_grid = {
        "n_estimators": [200, 300],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }

    # train and save the models
    trained_models = {}
    for model in models:
        if isinstance(model, type(models[0])):
            model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
            model.fit(x_train, y_train)
            trained_models["RandomForestClassifier"] = model.best_estimator_
            joblib.dump(model.best_estimator_, f"./{paths[0]}")
        else:
            model.fit(x_train, y_train)
            trained_models["LogisticRegression"] = model
            joblib.dump(model, f"./{paths[1]}")

    # generate and save each model's test set ROC curves
    plt.figure(figsize=(15, 8))
    for name, clf in trained_models.items():
        pos_class_pred_proba = clf.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pos_class_pred_proba)
        auc = np.round(roc_auc_score(y_test, pos_class_pred_proba), 2)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./{paths[2]}")


def feature_importance_plot(x_train: pd.DataFrame) -> None:
    """
    Generates a bar plot of the RandomForestClassifier's feature
    importances, and saves it to the 'results' folder

    Args:
        x_train: (pd.DataFrame) Train set feature matrix

    Returns:
        None
    """
    rfc = joblib.load(r"./models/rfc_model.pkl")
    importances = rfc.feature_importances_
    features = x_train.columns.tolist()
    (pd.DataFrame
     ({"importance": importances}, index=features)
     .sort_values("importance", ascending=False)
     .plot(kind="bar",
           figsize=(20, 7),
           title="Feature Importance",
           ylabel="Importance",
           legend=False)
     )
    plt.tight_layout()
    plt.savefig(r"./images/results/feature_importances.png")


def classification_report_image(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """
    Generates train and test set classification reports for
    each model, and saves them to the 'results' folder

    Args:
        x_train: (pd.DataFrame) Train set feature matrix
        x_test: (pd.DataFrame) Test set feature matrix
        y_train: (pd.Series) Train set target vector
        y_test: (pd.Series) Test set target vector

    Returns:
        None
    """
    rfc = joblib.load(r"./models/rfc_model.pkl")
    logr = joblib.load(r"./models/logistic_model.pkl")

    model_mapper = {
        "RandomForestClassifier": [rfc, "rf_results.png"],
        "LogisticRegression": [logr, "logistic_results.png"]
    }

    for name, lst in model_mapper.items():
        yhat_train = lst[0].predict(x_train)
        yhat_test = lst[0].predict(x_test)
        plt.figure(figsize=(5, 5))
        plt.text(0, 0.9, f"{name} Train Results:", fontproperties="monospace")
        plt.text(0, 0.55, classification_report(y_train, yhat_train), fontproperties="monospace")
        plt.text(0, 0.4, f"{name} Test Results:", fontproperties="monospace")
        plt.text(0, 0.05, classification_report(y_test, yhat_test), fontproperties="monospace")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"./images/results/{lst[1]}")
