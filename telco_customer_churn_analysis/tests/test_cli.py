import pytest
from typer.testing import CliRunner
from src.telco_customer_churn_analysis.cli import app
from unittest import mock

runner = CliRunner()

@mock.patch("src.telco_customer_churn_analysis.cli.exploratory_analysis")
@mock.patch("src.telco_customer_churn_analysis.cli.data_preprocessing")
def test_eda(mock_preprocess, mock_explore):
    mock_preprocess.return_value = "mock_df"
    mock_explore.return_value = None

    result = runner.invoke(app, ["eda"])

    assert result.exit_code == 0


@mock.patch("src.telco_customer_churn_analysis.cli.hypothesis_test")
@mock.patch("src.telco_customer_churn_analysis.cli.bin_df")
@mock.patch("src.telco_customer_churn_analysis.cli.data_preprocessing")
def test_hypothesis_tests_chi2_with_test(mock_preprocess, mock_bin, mock_hypo):
    mock_preprocess.return_value = "mock_df"
    mock_bin.return_value = "binned_df"

    result = runner.invoke(app, ["hypothesis-tests-chi2", "--test-or-new", "Test"])
    assert result.exit_code == 0


@mock.patch("src.telco_customer_churn_analysis.cli.joblib.load")
@mock.patch("src.telco_customer_churn_analysis.cli.get_model")
@mock.patch("src.telco_customer_churn_analysis.cli.bin_df")
@mock.patch("src.telco_customer_churn_analysis.cli.data_preprocessing")
@mock.patch("src.telco_customer_churn_analysis.cli.deploy_model")
def test_train_evaluate_deploy(mock_deploy, mock_preprocess, mock_bin, mock_get_model, mock_joblib):
    mock_preprocess.return_value = "mock_df"
    mock_bin.return_value = "mock_binned_df"
    mock_get_model.return_value = None
    mock_joblib.return_value = {'model_untrained': 'mock_model'}

    result = runner.invoke(app, ["train-evaluate-deploy"])
    assert result.exit_code == 0


@mock.patch("src.telco_customer_churn_analysis.cli.abc_test_assignment")
@mock.patch("src.telco_customer_churn_analysis.cli.predict_df")
@mock.patch("src.telco_customer_churn_analysis.cli.profit_curve_threshold")
@mock.patch("src.telco_customer_churn_analysis.cli.joblib.load")
@mock.patch("src.telco_customer_churn_analysis.cli.bin_df")
@mock.patch("src.telco_customer_churn_analysis.cli.generate_test_data")
def test_predict_with_best_profit_threshold(mock_gen_data, mock_bin, mock_joblib, mock_profit, mock_predict, mock_abc):
    mock_gen_data.return_value = "mock_df"
    mock_bin.return_value = "mock_binned"
    mock_profit.return_value = 0.6
    mock_predict.return_value = "predicted_df"
    mock_abc.return_value = "abc_df"
    mock_joblib.return_value = {'all_results': 'model_df', 'y_test': 'y_test'}

    result = runner.invoke(app, ["predict-with-best-profit-threshold"])
    assert result.exit_code == 0
    assert "Model deployed successfully!" in result.output


@mock.patch("src.telco_customer_churn_analysis.cli.local_explainer")
@mock.patch("src.telco_customer_churn_analysis.cli.global_explainer")
@mock.patch("src.telco_customer_churn_analysis.cli.joblib.load")
@mock.patch("src.telco_customer_churn_analysis.cli.abc_test_assignment")
@mock.patch("src.telco_customer_churn_analysis.cli.predict_df")
@mock.patch("src.telco_customer_churn_analysis.cli.bin_df")
@mock.patch("src.telco_customer_churn_analysis.cli.generate_test_data")
def test_predict_with_xai(mock_gen, mock_bin, mock_predict, mock_abc, mock_joblib, mock_global, mock_local):
    mock_gen.return_value = "mock_df"
    mock_bin.return_value = "mock_binned"
    mock_predict.return_value = "predicted_df"
    mock_abc.return_value = "abc_df"
    mock_joblib.return_value = {'X_train': 'train', 'X_test': 'test'}

    result = runner.invoke(app, ["predict-with-xai", "--global-xai", "--local-xai"])
    assert result.exit_code == 0
    assert "Prediction Results with ABC assignment" in result.output
