import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import io
import os
from flask import Flask
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

from src.telco_customer_churn_analysis.telco_customer_churn_analysis_app import (data_preprocessing_app, data_preprocessing, exploratory_analysis_app,
                                                                                 bin_df_app, hypothesis_test_app, train_evaluate_deploy_app,
                                                                                 predict_with_best_profit_threshold_app,
                                                                                 predict_with_xai_app, generate_test_data)

@pytest.fixture(autouse=True)
def suppress_show(monkeypatch):
    """Automatically suppress matplotlib's plt.show() during tests to prevent plots from displaying."""

    monkeypatch.setattr(plt, 'show', lambda: None)


@pytest.fixture(autouse=True)
def close_figures_after_test():
    yield
    plt.close('all')
    

@pytest.fixture
def sample_df():
    """Return a minimal sample Telco customer DataFrame for unit testing."""

    data = {
        'CustomerID': [1, 2],
        'City': ['CityA', 'CityB'],
        'Total Charges': ['100', ''],
        'Monthly Charges': [50, 70],
        'Tenure Months': [2, 3],
        'Churn Value': [0, 1],
        'CLTV': [2000, 3000],
        'Churn Score': [20, 80],
        'Gender': ['Male', 'Female'],
        'Senior Citizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Phone Service': ['Yes', 'No'],
        'Multiple Lines': ['No', 'Yes'],
        'Internet Service': ['DSL', 'Fiber'],
        'Online Security': ['No', 'Yes'],
        'Online Backup': ['Yes', 'No'],
        'Device Protection': ['Yes', 'No'],
        'Tech Support': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'Streaming TV': ['No', 'Yes'],
        'Streaming Movies': ['Yes', 'No'],
        'Contract': ['Month-to-Month', 'Two Year'],
        'Paperless Billing': ['Yes', 'No'],
        'Payment Method': ['Electronic', 'Mailed'],
        'Churn Reason': ['Competitor', 'Dissatisfaction']
    }
    return pd.DataFrame(data)


@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.kaggle_download')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.read_excel')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.col_replace', side_effect=lambda df, col, old, new: df.replace({col: {old: new}}))
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.null_rows', side_effect=lambda df, col=None: df[col].isna() if col else df.isna())
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.df_loc', side_effect=lambda df, mask, col: df.loc[mask, col])
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.df_head')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.df_aggfunc', side_effect=lambda df, func, col: df[col].value_counts() if func=='value_counts' else df[col].mean() if func=='mean' else df[col].median())
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.drop_labels', side_effect=lambda df, cols: df.drop(columns=cols, errors='ignore'))
def test_data_preprocessing_app(mock_drop, mock_agg, mock_head, mock_loc, mock_null, mock_col, mock_read, mock_kaggle, sample_df):
    """Test that data_preprocessing_app loads, cleans, and summarizes the dataset correctly."""
    
    mock_kaggle.return_value = 'fake_path.xlsx'
    mock_read.return_value = sample_df.copy()

    df, output_text = data_preprocessing_app()

    assert isinstance(df, pd.DataFrame)
    assert 'Total Charges' in df.columns
    assert 'CustomerID' not in df.columns
    assert 'Churn Value' in df.columns
    assert '--- Number of Missing Values by Column ---' in output_text
    assert '--- Number of Rows by Churn Value ---' in output_text


## exploratory_analysis_app tests

@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.os.makedirs')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.plt.savefig')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.plt.close')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.count_plot')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.histogram')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.heatmap')
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.bin_and_plot', side_effect=lambda *args, **kwargs: kwargs.get('df', args[2]))
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.df_aggfunc', side_effect=lambda df, func, col: df[col].value_counts())
@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.current_app', new_callable=MagicMock)
def test_exploratory_analysis_app(mock_app, mock_agg, mock_bin, mock_heat, mock_hist, mock_count, mock_close, mock_save, mock_makedirs, sample_df):
    """Test `exploratory_analysis_app()` generates plots and saves output image files."""
    
    mock_app.root_path = '/tmp'
    df, filenames = exploratory_analysis_app(sample_df.copy())

    assert isinstance(df, pd.DataFrame)
    assert isinstance(filenames, list)
    assert len(filenames) > 0
    for fname in filenames:
        assert fname.endswith('.png')


## bin_df_app tests

@patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.bin_and_plot', side_effect=lambda *args, **kwargs: kwargs.get('df', args[2]))
def test_bin_df_app(mock_bin, sample_df):
    """Test that `bin_df_app()` correctly bins tenure data and handles numeric conversion."""

    df_binned = bin_df_app(sample_df.copy(), show=False)

    assert 'Tenure Months' in df_binned.columns
    assert pd.api.types.is_numeric_dtype(df_binned['Tenure Months'])


## hypothesis_test_app tests

def test_hypothesis_test_app_test_data(capsys):
    """Test `hypothesis_test_app()` runs with preprocessed test data and produces valid output."""

    output = hypothesis_test_app(data_choice='Test')
    captured = capsys.readouterr()
    assert 'Use test data' in captured.out
    assert isinstance(output, str)
    assert len(output) > 0


def test_hypothesis_test_app_new_data(capsys):
    """Test `hypothesis_test_app()` runs with synthetic data and suppresses statistical warnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        output = hypothesis_test_app(data_choice='New')

    captured = capsys.readouterr()
    assert 'Use new data' in captured.out
    assert isinstance(output, str)
    assert len(output) > 0


def test_hypothesis_test_app_invalid_choice():
    """Test that `hypothesis_test_app()` raises ValueError for invalid dataset selection."""

    with pytest.raises(ValueError) as exc_info:
        hypothesis_test_app(data_choice='Invalid')

    assert 'Input invalid' in str(exc_info.value)


def test_hypothesis_test_app_with_specific_columns(capsys):
    """Test `hypothesis_test_app()` works when specific column names are provided."""

    df = data_preprocessing()
    col1, col2 = df.columns[0], df.columns[1]
    output = hypothesis_test_app(data_choice='Test', col1=col1, col2=col2)
    captured = capsys.readouterr()
    
    assert 'Use test data' in captured.out
    assert isinstance(output, str)
    assert len(output) > 0


def test_hypothesis_test_app_missing_columns(monkeypatch):
    """Test that `hypothesis_test_app()` raises ValueError when the DataFrame has no columns."""

    empty_df = pd.DataFrame()

    monkeypatch.setattr('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.generate_test_data', lambda: empty_df)
    monkeypatch.setattr('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.bin_df_app', lambda df: df)
    monkeypatch.setattr('src.telco_customer_churn_analysis.telco_customer_churn_analysis_app.chi_squared_test', lambda df, c1, c2: (None, None, None))

    with pytest.raises(ValueError) as exc_info:
        hypothesis_test_app(data_choice='New')
    
    assert 'Please select an dataframe column' in str(exc_info.value)


## train_evaluate_deploy_app tests

def test_train_evaluate_deploy_app(capsys):
    """Test the training, evaluation, and deployment process for expected console output."""

    output = train_evaluate_deploy_app()
    captured = capsys.readouterr()
    assert 'Start Training and Evaluate -- App' in captured.out
    assert isinstance(output, str)
    assert len(output) > 0


## predict_with_best_profit_threshold_app tests

def test_predict_with_best_profit_threshold_app_minimal(capsys):
    """Test profit threshold prediction using a provided DataFrame input."""

    df = generate_test_data()
    threshold, html = predict_with_best_profit_threshold_app(df=df, abc_assignment=False)
    captured = capsys.readouterr()
    assert isinstance(threshold, float)
    assert isinstance(html, str)
    assert 'Model deployed successfully!' in captured.out

def test_predict_with_best_profit_threshold_app_with_features(capsys):
    """Test profit threshold prediction when feature arguments are passed directly."""

    threshold, html = predict_with_best_profit_threshold_app(
        City='CityX',
        Gender='Female',
        Tenure_Months=12,
        Monthly_Charges=50.0
    )
    captured = capsys.readouterr()
    assert isinstance(threshold, float)
    assert isinstance(html, str)
    assert 'Features cleaned' in captured.out


## predict_with_xai_app tests

def test_predict_with_xai_app_basic(capsys):
    """Test XAI prediction function without XAI requests (default behavior)."""

    df = generate_test_data()
    predicted_html, global_xai_img, local_xai_img = predict_with_xai_app(df=df, threshold_input=0.5)
    captured = capsys.readouterr()
    
    assert isinstance(predicted_html, str)
    assert global_xai_img == ""
    assert local_xai_img == ""
    

def test_predict_with_xai_app_features_and_xai(capsys):
    """Test XAI prediction with both global and local explainability requests."""

    predicted_html, global_xai_img, local_xai_img = predict_with_xai_app(
        City='CityY',
        Gender='Male',
        Tenure_Months=24,
        global_xai=True,
        local_xai=True,
        index_local=0
    )
    captured = capsys.readouterr()
    assert isinstance(predicted_html, str)
    assert isinstance(global_xai_img, str)
    assert isinstance(local_xai_img, str)
