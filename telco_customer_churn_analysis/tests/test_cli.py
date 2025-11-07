import pytest
from typer.testing import CliRunner
from src.telco_customer_churn_analysis.cli import app
from unittest import mock
import pandas as pd

from src.telco_customer_churn_analysis.cli import hypothesis_tests_chi2

runner = CliRunner()

mock_df = pd.DataFrame({
    'A': ['x', 'y', 'x'],
    'B': ['u', 'v', 'u'],
    'C': [1, 2, 3]
})

@mock.patch("src.telco_customer_churn_analysis.cli.exploratory_analysis")
@mock.patch("src.telco_customer_churn_analysis.cli.data_preprocessing")
def test_eda(mock_preprocess, mock_explore):
    """Test that the 'eda' CLI command runs successfully."""

    mock_preprocess.return_value = "mock_df"
    mock_explore.return_value = None

    result = runner.invoke(app, ["eda"])

    assert result.exit_code == 0


@mock.patch('src.telco_customer_churn_analysis.cli.data_preprocessing', return_value=mock_df)
@mock.patch('src.telco_customer_churn_analysis.cli.bin_df', return_value=mock_df)
@mock.patch('src.telco_customer_churn_analysis.cli.hypothesis_test')
def test_hypothesis_tests_chi2_test_data(mock_hypothesis, mock_bin, mock_data):
    hypothesis_tests_chi2(data_choice='Test', col1='A', col2='B')
    mock_data.assert_called_once()
    mock_bin.assert_called_once_with(mock_df)
    mock_hypothesis.assert_called_once_with('Test', 'A', 'B')


@mock.patch('src.telco_customer_churn_analysis.cli.generate_test_data', return_value=mock_df)
@mock.patch('src.telco_customer_churn_analysis.cli.bin_df_app', return_value=mock_df)
@mock.patch('src.telco_customer_churn_analysis.cli.hypothesis_test')
def test_hypothesis_tests_chi2_new_data(mock_hypothesis, mock_bin_app, mock_generate):
    hypothesis_tests_chi2(data_choice='New', col1='A', col2='B')
    mock_generate.assert_called_once()
    mock_bin_app.assert_called_once_with(mock_df)
    mock_hypothesis.assert_called_once_with('New', 'A', 'B')


@mock.patch('src.telco_customer_churn_analysis.cli.data_preprocessing', return_value=mock_df)
@mock.patch('src.telco_customer_churn_analysis.cli.bin_df', return_value=mock_df)
@mock.patch('src.telco_customer_churn_analysis.cli.hypothesis_test')
def test_hypothesis_tests_chi2_default_columns(mock_hypothesis, mock_bin, mock_data):
    hypothesis_tests_chi2(data_choice='Test', col1=None, col2=None)
    mock_hypothesis.assert_called_once_with('Test', 'A', 'B')  # First two columns are used


def test_hypothesis_tests_chi2_invalid_data_choice():
    with pytest.raises(ValueError, match='Input invalid'):
        hypothesis_tests_chi2(data_choice='Invalid')


@mock.patch('src.telco_customer_churn_analysis.cli.data_preprocessing', return_value=pd.DataFrame({'A':[1]}))
@mock.patch('src.telco_customer_churn_analysis.cli.bin_df', return_value=pd.DataFrame({'A':[1]}))
def test_hypothesis_tests_chi2_insufficient_columns(mock_bin, mock_data):
    with pytest.raises(ValueError, match='Please select an dataframe column.'):
        hypothesis_tests_chi2(data_choice='Test')


@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("src.telco_customer_churn_analysis.cli.joblib.load")
@mock.patch("src.telco_customer_churn_analysis.cli.get_model")
@mock.patch("src.telco_customer_churn_analysis.cli.bin_df")
@mock.patch("src.telco_customer_churn_analysis.cli.data_preprocessing")
@mock.patch("src.telco_customer_churn_analysis.cli.deploy_model")
def test_train_evaluate_deploy(mock_deploy, mock_preprocess, mock_bin, mock_get_model, mock_joblib, mock_file):
    """Test that 'train-evaluate-deploy' CLI command executes successfully."""

    mock_preprocess.return_value = "mock_df"
    mock_bin.return_value = "mock_binned_df"
    mock_get_model.return_value = ("all_models", "all_results", "best_model", "X_train", "X_test", "y_test")
    mock_joblib.return_value = {
        'all_models': {'mock_model': 'model_obj'},
        'all_results': {'name': ['mock_model'], 'accuracy': [1]},
        'model_untrained': 'mock_model',
        'X_train': 'X_train',
        'X_test': 'X_test',
        'y_test': 'y_test'
        }
    
    mock_deploy.return_value = None

    result = runner.invoke(app, ["train-evaluate-deploy"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Get Model" in result.output


@mock.patch("src.telco_customer_churn_analysis.cli.abc_test_assignment")
@mock.patch("src.telco_customer_churn_analysis.cli.predict_df")
@mock.patch("src.telco_customer_churn_analysis.cli.profit_curve_threshold")
@mock.patch("src.telco_customer_churn_analysis.cli.joblib.load")
@mock.patch("src.telco_customer_churn_analysis.cli.bin_df_app")
@mock.patch("src.telco_customer_churn_analysis.cli.generate_test_data")
def test_predict_with_best_profit_threshold(mock_gen_data, mock_bin, mock_joblib, mock_profit, mock_predict, mock_abc):
    """Test that 'predict-with-best-profit-threshold' CLI command runs and deploys the model."""

    mock_gen_data.return_value = "mock_df"
    mock_bin.return_value = "mock_binned"
    mock_profit.return_value = 0.6
    mock_predict.return_value = "predicted_df"
    mock_abc.return_value = "abc_df"
    mock_joblib.return_value = {'all_results': 'model_df', 'y_test': 'y_test'}

    result = runner.invoke(app, ["predict-with-best-profit-threshold"], catch_exceptions=False)
    assert result.exit_code == 0
    
    if "Warning: Could not load 'model_results.pkl'" in result.output:
        assert "Please run `comparative_models`" in result.output

    else:
        assert "Model deployed successfully!" in result.output


@mock.patch("src.telco_customer_churn_analysis.cli.local_explainer")
@mock.patch("src.telco_customer_churn_analysis.cli.global_explainer")
@mock.patch("src.telco_customer_churn_analysis.cli.joblib.load")
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("src.telco_customer_churn_analysis.cli.abc_test_assignment")
@mock.patch("src.telco_customer_churn_analysis.cli.predict_df")
@mock.patch("src.telco_customer_churn_analysis.cli.bin_df_app")
@mock.patch("src.telco_customer_churn_analysis.cli.generate_test_data")
def test_predict_with_xai(mock_gen, mock_bin, mock_predict, mock_abc, mock_open_file, mock_joblib, mock_global, mock_local):
    """Test that 'predict-with-xai' CLI command executes with global and local XAI options."""
    
    mock_gen.return_value = "mock_df"
    mock_bin.return_value = "mock_binned"
    mock_predict.return_value = "predicted_df"
    mock_abc.return_value = "abc_df"

    mock_joblib.return_value = {
        'X_train': pd.DataFrame({'pred_proba': [0.2, 0.4, 0.6, 0.8], 'real': [1, 0, 1, 0]}),
        'X_test': pd.DataFrame({'pred_proba': [0.2, 0.4], 'real': [1, 0]})}

    result = runner.invoke(app, ["predict-with-xai", "--global-xai", "--local-xai"], catch_exceptions=False)
    assert result.exit_code == 0