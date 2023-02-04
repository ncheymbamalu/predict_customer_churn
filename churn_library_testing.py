"""
Testing for churn_library.py

author: Nchey Mbamalu
date: Feb 2023
"""
import churn_library_logging as cll


def test_import_data():
    """test the 'import_data' function"""
    cll.import_data(r"./data/bank_data.csv")


def test_perform_eda():
    """test the 'perform_eda' function"""
    df_ = cll.import_data(r"./data/bank_data.csv")
    cll.perform_eda(df_, r"./images/eda")


def test_encoder_helper():
    """test the 'encoder_helper' function"""
    df_ = cll.import_data(r"./data/bank_data.csv")
    cll.encoder_helper(df_)


def test_perform_feature_engineering():
    """test the 'perform_feature_engineering' function"""
    df_ = cll.import_data(r"./data/bank_data.csv")
    df_encoded = cll.encoder_helper(df_)
    cll.perform_feature_engineering(df_encoded)


def test_train_models():
    """test the 'train_models' function"""
    df_ = cll.import_data(r"./data/bank_data.csv")
    df_encoded = cll.encoder_helper(df_)
    x_train, x_test, y_train, y_test = cll.perform_feature_engineering(df_encoded)
    cll.train_models(x_train, x_test, y_train, y_test)


def test_feature_importance_plot():
    """test the 'feature_importance_plot' function"""
    df_ = cll.import_data(r"./data/bank_data.csv")
    df_encoded = cll.encoder_helper(df_)
    x_train, _, _, _ = cll.perform_feature_engineering(df_encoded)
    cll.feature_importance_plot(x_train)


def test_classification_report_image():
    """test the 'classification_report_image' function"""
    df_ = cll.import_data(r"./data/bank_data.csv")
    df_encoded = cll.encoder_helper(df_)
    x_train, x_test, y_train, y_test = cll.perform_feature_engineering(df_encoded)
    cll.classification_report_image(x_train, x_test, y_train, y_test)
