import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest import mock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


try:
  from xgboost import XGBClassifier
except ImportError:
  class XGBClassifier: pass

try:
  from lightgbm import LGBMClassifier
except ImportError:
  class LGBMClassifier: pass

from src.telco_customer_churn_analysis.model_xai import (model_global_explainer, model_local_explainer)


@pytest.fixture
def sample_df():
    """Provides a sample DataFrame for X_train/X_testwith mixed types."""
    np.random.seed(123)

    n_rows = 1100
    data = {
        'A': np.random.rand(n_rows),
        'B': np.random.choice(['x', 'y', 'z'], n_rows),
        'C': np.random.choice([0, 1], n_rows),
        'Target': np.random.choice([0, 1], n_rows)
    }
    return pd.DataFrame(data)

@pytest.fixture
def create_mock_pipeline_steps():
    """Return mock preprocessor and classifier objects."""
    n_rows = 1100
    n_features = 5
    mock_preprocessor = mock.MagicMock(spec=ColumnTransformer)
    mock_preprocessor.get_feature_names_out.return_value = ['A', 'B_x', 'B_y', 'B_z', 'C']
    mock_preprocessor.transform.return_value = np.random.rand(n_rows, n_features)

    mock_tree_classifier = mock.MagicMock(spec=DecisionTreeClassifier)
    mock_internal_tree = mock.MagicMock()
    num_nodes = 3
    mock_internal_tree.node_count = num_nodes
    mock_internal_tree.children_left = np.array([-1] * num_nodes)
    mock_internal_tree.children_right = np.array([-1] * num_nodes)
    mock_internal_tree.feature = np.array([0] * num_nodes)
    mock_internal_tree.threshold = np.array([0.0] * num_nodes)
    mock_internal_tree.value = np.array([[0.0, 0.0]] * num_nodes)
    mock_internal_tree.n_features = 5
    mock_tree_classifier.tree_ = mock_internal_tree
    mock_tree_classifier.n_features_ = 5

    mock_linear_classifier = mock.MagicMock(spec=LogisticRegression)
    mock_linear_classifier.predict_proba.return_value = np.array([[0.5, 0.5]] * n_rows)

    return mock_preprocessor, mock_tree_classifier, mock_linear_classifier


@pytest.fixture
def tree_model_pipeline(create_mock_pipeline_steps):
    """Provide a mock Pipeline with Tree-based classifier."""
    mock_preprocessor, mock_tree_classifier, _ = create_mock_pipeline_steps
    pipeline = Pipeline(steps=[
        ('preprocessor', mock_preprocessor),
        ('classifier', mock_tree_classifier)
    ])

    return pipeline


@pytest.fixture
def linear_model_pipeline(create_mock_pipeline_steps):
    """Provides a mock Pipeline with a non-Tree-based classifier."""
    mock_preprocessor, _ , mock_linear_classifier = create_mock_pipeline_steps
    pipeline = Pipeline(steps=[
        ('preprocessor', mock_preprocessor),
        ('classifier', mock_linear_classifier)
    ])

    return pipeline


@pytest.fixture(autouse=True)
def mock_external_libs():
    """Mock SHAP and Matplotlib for all tests in this file."""
    n_rows = 1100
    n_features = 5

    with mock.patch.dict('sys.modules', {'shap': mock.MagicMock(), 'matplotlib.pyplot': mock.MagicMock()}):
        mock_shap = mock.MagicMock()
        mock_shap.TreeExplainer = mock.MagicMock()
        mock_shap.KernelExplainer = mock.MagicMock()

        mock_shap.TreeExplainer.return_value.shap_values.return_value = [np.random.rand(n_rows, n_features), np.random.rand(n_rows, n_features)]
        mock_shap.TreeExplainer.return_value.expected_value = np.array([0.5, 0.5])

        mock_shap.KernelExplainer.return_value.shap_values.return_value = [np.random.rand(n_rows, n_features), np.random.rand(n_rows, n_features)]
        mock_shap.KernelExplainer.return_value.expected_value = np.array([0.5, 0.5])

        mock_shap.kmeans.return_value.data = np.random.rand(100, n_features)

        mock_shap.summary_plot = mock.MagicMock()
        mock_shap.waterfall_plot = mock.MagicMock()

        mock_shap.Explanation = mock.MagicMock()

        sys_modules = {'shap': mock_shap}

        mock_plt = mock.MagicMock()
        mock_plt.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
        sys_modules['matplotlib.pyplot'] = mock_plt

        with mock.patch.dict('sys.modules', sys_modules):
            with mock.patch('src.telco_customer_churn_analysis.model_xai.DecisionTreeClassifier', DecisionTreeClassifier), \
            mock.patch('src.telco_customer_churn_analysis.model_xai.RandomForestClassifier', RandomForestClassifier), \
            mock.patch('src.telco_customer_churn_analysis.model_xai.GradientBoostingClassifier', GradientBoostingClassifier), \
            mock.patch('src.telco_customer_churn_analysis.model_xai.XGBClassifier', XGBClassifier), \
            mock.patch('src.telco_customer_churn_analysis.model_xai.LGBMClassifier', LGBMClassifier):
                yield mock_shap, mock_plt


## test model global explainer

class TestModelGlobalExplainer:
    @staticmethod
    def test_model_global_explainer_raises_value_error_if_not_pipeline(sample_df):
        """Test ValueError when model is not a Pipeline."""
        with pytest.raises(ValueError, match="The 'model' must be a sklearn.pipeline.Pipeline object"):
            model_global_explainer(mock.MagicMock(), sample_df, sample_df)

    
    @staticmethod
    def test_model_global_explainer_raises_value_error_if_missing_steps(sample_df):
        """Test ValueError when required steps are missing."""
        bad_pipeline = Pipeline(steps=[('only_step', mock.MagicMock())])
        with pytest.raises(ValueError, match="Pipeline must contain steps named 'preprocessor' and 'classifier'"):
            model_global_explainer(bad_pipeline, sample_df, sample_df)


    @staticmethod
    def test_model_global_explainer_raises_attribute_error_on_feature_names(tree_model_pipeline, sample_df):
        """Test AttributeError when preprocessor lacks 'get_feature_names_out()'."""
        tree_model_pipeline.named_steps['preprocessor'].get_feature_names_out.side_effect = AttributeError()
        with pytest.raises(AttributeError, match="Failed to retrieve feature names"):
            model_global_explainer(tree_model_pipeline, sample_df, sample_df)

    
    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.model_xai.shap')
    def test_model_global_explainer_uses_tree_explainer(mock_shap, tree_model_pipeline, sample_df):
        """Test TreeExplainer path is correctly used for tree-based models."""
        mock_tree_explainer = mock_shap.TreeExplainer
        model_global_explainer(tree_model_pipeline, sample_df, sample_df)

        mock_tree_explainer.assert_called_once()
        mock_shap.KernelExplainer.assert_not_called()
        mock_shap.summary_plot.assert_called_once()


    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.model_xai.shap')
    def test_model_global_explainer_uses_kernel_explainer(mock_shap, linear_model_pipeline, sample_df):
        """Test KernelExplainer path is correctly used for non-tree models."""
        mock_kernel_explainer = mock_shap.KernelExplainer
        model_global_explainer(linear_model_pipeline, sample_df, sample_df)

        mock_kernel_explainer.assert_called_once()
        mock_shap.TreeExplainer.assert_not_called()
        mock_shap.kmeans.assert_called_once()
        mock_shap.summary_plot.assert_called_once()


    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.model_xai.shap')
    def test_model_global_explainer_kernel_explainer_limits_test_set(mock_shap, linear_model_pipeline, sample_df):
        """Test KernelExplainer limits the test set to 1000 samples for performance."""
        large_df = pd.concat([sample_df] * 12).reset_index(drop=True)
        model_global_explainer(linear_model_pipeline, sample_df, large_df, random_state=1)

        call_args = mock_shap.KernelExplainer.return_value.shap_values.call_args[0]
        X_test_for_shap_arg = call_args[0]

        large_transformed = np.random.rand(1200, 5)
        linear_model_pipeline.named_steps['preprocessor'].transform.side_effect = lambda X: large_transformed if len(X) > 1000 else np.random.rand(len(X), 5)

        model_global_explainer(linear_model_pipeline, sample_df, large_df, random_state=1)

        call_args = mock_shap.KernelExplainer.return_value.shap_values.call_args[0]
        X_test_for_shap_arg = call_args[0]

        assert X_test_for_shap_arg.shape[0] == 1000



## test model local explainer

class TestModelLocalExplainer:
    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.model_xai.shap.TreeExplainer')
    def test_local_model_explainer_raises_value_error_if_index_out_of_bounds(mock_tree_explainer, tree_model_pipeline, sample_df):
        """Test ValueError when the index is not valid for X_test."""
        mock_tree_explainer.return_value = mock.MagicMock()

        X_test = sample_df.loc[: 10]

        mock_preprocessor = tree_model_pipeline.named_steps['preprocessor']

        mock_preprocessor.transform.side_effect = [
            np.random.rand(1100, 5),
            np.random.rand(11, 5),
            np.random.rand(1100, 5),
            np.random.rand(11, 5)
        ]

        index = 11
        error_match = f"ValueError: Index {index} is out of bounds. Must be between 0 and {len(X_test) - 1}."
        with pytest.raises(ValueError, match=error_match):
            model_local_explainer(tree_model_pipeline, sample_df, X_test, index=index)

        index = -1
        error_match = f"ValueError: Index {index} is out of bounds. Must be between 0 and {len(X_test) - 1}."
        with pytest.raises(ValueError, match=error_match):
            model_local_explainer(tree_model_pipeline, sample_df, X_test, index=index)


    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.model_xai.shap')
    def test_model_local_explainer_uses_tree_explainer_and_waterfall(mock_shap, tree_model_pipeline, sample_df):
        """Test TreeExplainer path and correct waterfall plot call."""
        mock_tree_explainer = mock_shap.TreeExplainer
        model_local_explainer(tree_model_pipeline, sample_df, sample_df, index=5)

        mock_tree_explainer.assert_called_once()
        mock_shap.KernelExplainer.assert_not_called()
        mock_shap.waterfall_plot.assert_called_once()

        call_args = mock_tree_explainer.return_value.shap_values.call_args[0]
        X_test_instance_arg = call_args[0]

        assert X_test_instance_arg.shape == (1, 5)


    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.model_xai.shap')
    def test_model_local_explainer_uses_kernel_explainer_and_waterfall(mock_shap, linear_model_pipeline, sample_df):
        """Test KernelExplainer path and correct waterfall plot call."""
        mock_kernel_explainer = mock_shap.KernelExplainer
        model_local_explainer(linear_model_pipeline, sample_df, sample_df, index=5)

        mock_kernel_explainer.assert_called_once()
        mock_shap.TreeExplainer.assert_not_called()
        mock_shap.kmeans.assert_called_once()
        mock_shap.waterfall_plot.assert_called_once()

        call_args = mock_kernel_explainer.return_value.shap_values.call_args[0]
        X_test_instance_arg = call_args[0]

        assert X_test_instance_arg.shape == (1, 5)

        kernel_explainer_call = mock_kernel_explainer.call_args[0]
        model_predict_proba_arg = kernel_explainer_call[0]

        assert model_predict_proba_arg == linear_model_pipeline.named_steps['classifier'].predict_proba
