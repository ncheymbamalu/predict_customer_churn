"""
Logging for churn_library.py

author: Nchey Mbamalu
date: Feb 2023
"""
from typing import Tuple
import logging
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report


logging.basicConfig(
    filename=r"./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s"
)


def import_data(path: str) -> pd.DataFrame:
    """
    Reads in path as a Pandas DataFrame

    Args:
        path: (str) path to the input data

    Returns:
        df_: (pd.DataFrame) Pandas DataFrame
    """
    filename = path.split("/")[-1]
    try:
        df_ = pd.read_csv(path, index_col=0)
        target_encoder = {"Existing Customer": 0, "Attrited Customer": 1}
        df_ = (df_
               .assign(Attrition_Flag=df_["Attrition_Flag"].map(target_encoder))
               .rename({"Attrition_Flag": "Churn"}, axis=1)
               .drop("CLIENTNUM", axis=1)
               .copy(deep=True)
               )
        logging.info("import_data - SUCCESS: %s read in", filename)
    except FileNotFoundError as err:
        logging.error("import_data - ERROR: %s not found", filename)
        raise err

    try:
        assert df_.shape[0] > 0
        assert df_.shape[1] > 0
        return df_
    except AssertionError as err:
        logging.info("import_data - ERROR: %s contains no data", filename)
        raise err


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
    image_files = [
        "churn_distribution.png",
        "customer_age_distribution.png",
        "heatmap.png",
        "marital_status_distribution.png",
        "total_transaction_distribution.png"
    ]
    try:
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
            if col in ["Churn", "Customer_Age"]:
                df_[col].hist()
                plt.savefig(f"{path}/{col.lower()}_distribution.png")
            elif col == "Marital_Status":
                df_[col].value_counts(normalize=True).plot(kind="bar")
                plt.savefig(f"{path}/{col.lower()}_distribution.png")
            else:
                sns.histplot(df_[col], stat="density", kde=True)
                plt.savefig(f"{path}/total_transaction_distribution.png")
        time.sleep(10)
        for file in image_files:
            assert os.path.isfile(f"{path}/{file}")
            logging.info("perform_eda - SUCCESS: %s saved", file)
    except AssertionError as err:
        logging.error("perform_eda - ERROR: EDA figures not saved")
        raise err


def encoder_helper(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that coverts each categorical feature to
    an encoded feature, with the average churn for each category

    Args:
        dataframe: (pd.DataFrame) Pandas DataFrame

    Returns:
        df_: (pd.DataFrame) Pandas DataFrame with encoded columns
    """
    encoded_cols = [
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"
    ]
    try:
        df_ = dataframe.copy(deep=True)
        cat_cols = df_.drop("Churn", axis=1).select_dtypes("object").columns
        for col in cat_cols:
            categories = df_.groupby(col)["Churn"].mean().index.tolist()
            churn_averages = df_.groupby(col)["Churn"].mean().tolist()
            df_[f"{col}_Churn"] = df_[col].map(dict(zip(categories, churn_averages)))
        assert [col for col in df_.columns if col not in dataframe.columns] == encoded_cols
        assert df_.shape == (dataframe.shape[0], dataframe.shape[1] + len(encoded_cols))
        logging.info("encoder_helper - SUCCESS: categorical features encoded")
        return df_
    except AssertionError as err:
        logging.error("encoder_helper - ERROR: categorical features not encoded")
        raise err


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
    try:
        features = df_.drop("Churn", axis=1).select_dtypes("number").columns.tolist()
        x_train, x_test, y_train, y_test = train_test_split(
            df_[features],
            df_["Churn"],
            test_size=0.3,
            random_state=42
        )
        assert x_train.shape[0] + x_test.shape[0] == df_.shape[0]
        logging.info("perform_feature_engineering - SUCCESS: train and test sets generated")
        return x_train, x_test, y_train, y_test
    except AssertionError as err:
        logging.error("perform_feature_engineering - ERROR: train and test sets not generated")
        raise err


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
        r"models/rfc_model.pkl",
        r"models/logistic_model.pkl",
        r"images/results/roc_curve_result.png"
    ]
    try:
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

        # ROC curves
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
        time.sleep(270)
        for path in paths:
            assert os.path.isfile(f"./{path}")
            logging.info("train_models - SUCCESS: %s saved", path.split("/")[-1])
    except AssertionError as err:
        logging.error("train_models - ERROR: models and ROC curves not saved")
        raise err


def feature_importance_plot(x_train: pd.DataFrame) -> None:
    """
    Generates a bar plot of the RandomForestClassifier's feature
    importances, and saves it to the 'results' folder

    Args:
        x_train: (pd.DataFrame) Train set feature matrix

    Returns:
        None
    """
    fi_plot = "feature_importances.png"
    try:
        rfc = joblib.load(r"./models/rfc_model.pkl")
        (pd.DataFrame
         ({"importance": rfc.feature_importances_},
          index=x_train.columns.tolist()
          )
         .sort_values("importance", ascending=False)
         .plot(kind="bar",
               figsize=(20, 7),
               title="Feature Importance",
               ylabel="Importance",
               legend=False)
         )
        plt.tight_layout()
        plt.savefig(f"./images/results/{fi_plot}")
        time.sleep(10)
        assert os.path.isfile(f"./images/results/{fi_plot}")
        logging.info("feature_importance_plot - SUCCESS: %s saved", fi_plot)
    except AssertionError as err:
        logging.error("feature_importance_plot - ERROR: %s not saved", fi_plot)
        raise err


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
    reports = ["logistic_results.png", "rf_results.png"]
    try:
        rfc = joblib.load(r"./models/rfc_model.pkl")
        logr = joblib.load(r"./models/logistic_model.pkl")

        models = {
            "RandomForestClassifier": [rfc, "rf_results.png"],
            "LogisticRegression": [logr, "logistic_results.png"]
        }

        for name, lst in models.items():
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
        time.sleep(10)
        for report in reports:
            assert os.path.isfile(f"./images/results/{report}")
            logging.info("classification_report_image - SUCCESS: %s saved", report)
    except AssertionError as err:
        logging.error("classification_report_image - ERROR: Classification reports not saved")
        raise err


if __name__ == "__main__":
    df = import_data(r"./data/bank_data.csv")
    perform_eda(df, r"./images/eda")
    df_encoded = encoder_helper(df)
    Xtrain, Xtest, ytrain, ytest = perform_feature_engineering(df_encoded)
    train_models(Xtrain, Xtest, ytrain, ytest)
    feature_importance_plot(Xtrain)
    classification_report_image(Xtrain, Xtest, ytrain, ytest)
