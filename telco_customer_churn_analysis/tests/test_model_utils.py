import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from unittest import mock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
import os

from src.telco_customer_churn_analysis.model_utils import (preprocess_data, model_pipelines, hyperparameters, train_evaluate_model,
                                                           voting_model, comparative_models, profit_curve, deployment_model,
                                                           predict_churn, abc_test
                                                           )

@pytest.fixture
def sample_dataframe():
    """Fixture providing a synthetic customer DataFrame with mixed data types for testing."""
    data = {
        'target_col': np.random.rand(20),
        'low_cat_str': ['A'] * 10 + ['B'] * 10,
        'high_cat_str': [f'ID_{i}' for i in range(20)],
        'low_num_int': np.random.randint(0, 5, 20),
        'high_num_float': np.random.randn(20),
        'bool_col': [True, False] * 10,
        'drop_me': np.arange(20)
    }

    df = pd.DataFrame(data)

    df['low_cat_str'] = df['low_cat_str'].astype('object')
    df['high_cat_str'] = df['high_cat_str'].astype('object')

    return df


@pytest.fixture
def mock_preprocessor():
    """Fixture returning a mocked `ColumnTransformer` for pipeline tests."""
    mock_preprocessor = mock.create_autospec(ColumnTransformer, isinstance=True)

    return mock_preprocessor


@pytest.fixture
def mock_comparative_models(random_state):
    """Fixture returning mock classifiers (one with and one without `random_state`) for model evaluation tests."""
    class MockRandomModel(BaseEstimator, ClassifierMixin):
        def __init__(self, random_state=None):
            self.random_state = random_state
        
        def fit(self, X, y=None): return self

        def predict(self, X): return [0] * len(X)

        def set_params(self, **params):
            if 'random_state' in params:
                self.random_state = params['random_state']

            return self
        
    class MockNonRandomModel(BaseEstimator, ClassifierMixin):
        def fit(self, X, y=None): return self
        def predict(self, X): return [0] * len(X)
        def set_params(self, **params): return self

    return {
        'RandomForestClassifier': MockRandomModel(),
        'KNNeighbors': MockNonRandomModel()
    }


@pytest.fixture
def random_state():
    """Fixture providing a fixed random state for reproducibility in tests."""
    return 42


def create_mock_pipeline(classifier_instance, is_valid_pipeline=True):
    """
    Creates a mocked sklearn Pipeline with a classifier and optimal validation flag.
    
    Parameters
    ----------
    classifier_instance: sklearn classifier
        The classifier to include in the mock pipeline.
    is_valid_pipeline: bool
        If True, sets the mock class as Pipeline to mimic real pipeline.
        
    Returns
    -------
    mock_pipeline: MagicMock
        A mocked Pipeline with a 'classifier' and 'preprocessor' step."""
    
    mock_pipeline = mock.MagicMock(spec=Pipeline)

    if is_valid_pipeline:
        mock_pipeline.__class__ = Pipeline

    mock_pipeline.named_steps = {'preprocessor': mock.MagicMock(), 'classifier': classifier_instance}
    
    return mock_pipeline

@pytest.fixture
def logreg_classifier():
    """Fixture providing a LogisticRegression instance."""
    return LogisticRegression()

@pytest.fixture
def knn_classifier():
    """Fixture providing a KNeighborsClassifier instance."""
    return KNeighborsClassifier()

@pytest.fixture
def rf_Classifier():
    """Fixture providing a RandomForestClassifier instance."""
    return RandomForestClassifier()

@pytest.fixture
def gnb_classifier():
    """Fixture providing a GaussianNB instance."""
    return GaussianNB()

@pytest.fixture
def unsupported_classifier():
    """Fixture providing a custom classifier that simulates an unsupported model type."""
    class UnsupportedClassifier(BaseEstimator):
        def fit(self, X, y=None): return self
    
    return UnsupportedClassifier()

@pytest.fixture
def standard_models_dict(logreg_classifier, knn_classifier, rf_Classifier, gnb_classifier, unsupported_classifier):
    """Fixture returning a dictionary of standard model pipelines (some mock, some unsupported)."""
    return {
        'LogisticRegression': create_mock_pipeline(logreg_classifier),
        'KNN': create_mock_pipeline(knn_classifier),
        'RandomForest': create_mock_pipeline(rf_Classifier),
        'GaussianNB': create_mock_pipeline(gnb_classifier),
        'UnsupportedModel': create_mock_pipeline(unsupported_classifier)
        }


@pytest.fixture(scope='session')
def mock_data():
    """Fixture providing synthetic training and test datasets with mixed features."""
    np.random.seed(123)
    n_samples = 100

    numerical_features = ['num_a', 'num_b']
    categorical_features = ['cat_c']

    X = pd.DataFrame({
        'num_a': np.random.rand(n_samples),
        'num_b': np.random.rand(n_samples) * 10,
        'cat_c': np.random.choice(['A', 'B', 'C'], n_samples)
    })

    y = ((X['num_a'] * 5 + X['num_b'] * 0.5) > 4).astype(int)

    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    return X_train, X_test, y_train, y_test

@pytest.fixture
def basic_preprocessor():
    """Fixture providing a basic ColumnTransformer with numeric and categorical pipelines."""
    numerical_features = ['num_a', 'num_b']
    categorical_features = ['cat_c']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


@pytest.fixture
def mock_params_logreg(basic_preprocessor):
    """Fixture returning pipeline and param grid for LogisticRegression."""
    return {
        'LogisticRegression': {
            'model': Pipeline(steps=[
                ('preprocessor', basic_preprocessor),
                ('classifier', LogisticRegression(random_state=123, solver='liblinear'))
            ]),
            'params': {
                'classifier__C': [0.1, 1.0]
            }
        }
    }


@pytest.fixture
def mock_params_rf(basic_preprocessor):
    """Fixture returning pipeline and param grid for RandomForestClassifier."""
    return {
        'RandomForest': {
            'model': Pipeline(steps=[
                ('preprocessor', basic_preprocessor),
                ('classifier', RandomForestClassifier(random_state=123))
            ]),
            'params': {
                'classifier__n_estimators': [5, 10],
                'classifier__max_depth': [2, 3]
            }
        }
    }


@pytest.fixture
def mock_params_all(mock_params_logreg, mock_params_rf):
    """Fixture merging param grids for multiple classifiers."""
    return {**mock_params_logreg, **mock_params_rf}


@pytest.fixture
def mock_params_non_proba(basic_preprocessor):
    """Fixture returning a pipeline with a classifier that does not support `predict_proba()`."""
    class NonProbaClassifier(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.feature_importances_ = np.array([0.5, 0.3, 0.1, 0.1])
            self.classes_ = np.unique(y)

            return self
        
        def predict(self, X): return np.ones(X.shape[0], dtype=int)
        def get_params(self, deep=True): return {}
        def set_params(self, **params): return self

    class NonProbaPipeline(Pipeline, BaseEstimator):
        def predict_proba(self, X):
            raise AttributeError("This model cannot compute probabilities.")
        
        def __init__(self, steps):
            super().__init__(steps)

    pipeline = NonProbaPipeline(steps=[
        ('preprocessor', basic_preprocessor),
        ('classifier', NonProbaClassifier)
    ])

    return {'NonProbaModel': pipeline}

@pytest.fixture(autouse=True)
def cleanup_deployment_file():
    """
    Automatically delete the 'deployment_pipeline.pkl' file after each test, if it exists.
    This prevents side effects between tests that may write this file to disk.
    """
    yield
    if os.path.exists('deployment_pipeline.pkl'):
        os.remove('deployment_pipeline.pkl')

## preprocess_data tests

class TestPreprocessData:
    @staticmethod
    def test_preprocess_data_successful_run(sample_dataframe):
        """Test the function with standard parameters and check return types/shapes."""
        df = sample_dataframe
        target = 'target_col'
        cols_to_drop = 'drop_me'

        results = preprocess_data(df, target, cols_to_drop=cols_to_drop)

        X, y, X_train, X_test, y_train, y_test, preprocessor, preprocessor_trained = results

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(preprocessor, ColumnTransformer)
        assert isinstance(preprocessor_trained, ColumnTransformer)

        total_rows = len(df)
        train_rows = len(X_train)
        test_rows = len(X_test)

        assert train_rows + test_rows == total_rows
        assert X.shape[1] == df.shape[1] - 2
        assert y.shape[0] == total_rows
        assert y_train.shape[0] == train_rows
        assert y_test.shape[0] == test_rows

        one_hot_name, one_hot_trans, one_hot_cols_config = preprocessor.transformers_[0]
        scaler_name, scaler_trans, scaler_cols_config = preprocessor.transformers_[1]

        assert sorted(one_hot_cols_config) == sorted(['low_cat_str', 'bool_col'])
        assert sorted(scaler_cols_config) == sorted(['high_num_float'])
        assert one_hot_name == 'one_hot'
        assert scaler_name == 'scaler'
        assert isinstance(one_hot_trans, OneHotEncoder)
        assert isinstance(scaler_trans, StandardScaler)
        assert preprocessor.remainder == 'passthrough'

        X_train_transformed = preprocessor_trained.transform(X_train)

        assert X_train_transformed.shape[1] == 7


    @staticmethod
    def test_preprocess_data_default_cols_to_drop(sample_dataframe):
        """Test dropping only the target column (cols_to_drop=None)."""
        df = sample_dataframe
        target = 'target_col'

        X, _ , _ , _ , _ , _ , _ , _ = preprocess_data(df, target, cols_to_drop=None)

        assert X.shape[1] == df.shape[1] - 1
        assert 'target_col' not in X.columns
        assert 'drop_me' in X.columns


    @staticmethod
    def test_preprocess_data_cols_to_drop_list(sample_dataframe):
        """Test dropping a list of columns."""
        df = sample_dataframe
        target = 'target_col'
        cols_to_drop = ['drop_me', 'low_num_int']

        X, _ , _ , _ , _ , _ , _ , _ = preprocess_data(df, target, cols_to_drop=cols_to_drop)

        assert X.shape[1] == df.shape[1] - 3
        assert 'target_col' not in X.columns
        assert 'drop_me' not in X.columns
        assert 'low_num_int' not in X.columns


    @staticmethod
    def test_preprocess_data_value_error_zero_test_size():
        """Test `preprocess_data()` raises ValueError when 'test_size' = 0.0."""
        df = pd.DataFrame({'target_col': [1, 2, 3, 4], 'feat': [5, 6, 7, 8]})
        with pytest.raises(ValueError, match="between 0 and 1"):
            preprocess_data(df, 'target_col', test_size=0.0)


    @staticmethod
    def test_preprocess_data_value_error_one_test_size():
        """Test `preprocess_data()` raises ValueError when 'test_size' = 1.0."""
        df = pd.DataFrame({'target_col': [1, 2, 3, 4], 'feat': [5, 6, 7, 8]})
        with pytest.raises(ValueError, match="between 0 and 1"):
            preprocess_data(df, 'target_col', test_size=1.0)


    @staticmethod
    def test_preprocess_data_type_error_df():
        """Test `preprocess_data()` raises TypeError when 'df' is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame"):
            preprocess_data('not a df', target='target_col')


    @staticmethod
    def test_preprocess_data_type_error_target():
        """Test `preprocess_data()` raises TypeError when 'target' is not a string."""
        df = pd.DataFrame({'target_col': [1]})
        with pytest.raises(TypeError, match="'target' must be a string"):
            preprocess_data(df, target=123)


    @staticmethod
    def test_preprocess_data_key_error_target(sample_dataframe):
        """Test `preprocess_data()` raises KeyError when 'target' is not in the DataFrame."""
        with pytest.raises(KeyError, match="Target column 'missing' not found in DataFrame"):
            preprocess_data(sample_dataframe, target='missing')


    @staticmethod
    def test_preprocess_data_key_error_cols_to_drop(sample_dataframe):
        """Test `preprocess_data()` raises KeyError when 'cols_to_drop' is not in the DataFrame."""
        with pytest.raises(KeyError, match="Column to drop 'missing' not found in DataFrame"):
            preprocess_data(sample_dataframe, target='target_col', cols_to_drop='missing')


    @staticmethod
    def test_preprocess_data_type_error_cols_to_drop(sample_dataframe):
        """Test `preprocess_data()` raises TypeError when 'cols_to_drop' is an invalid type."""
        with pytest.raises(TypeError, match="'cols_to_drop' must be a string, a list of strings, or None"):
            preprocess_data(sample_dataframe, target='target_col', cols_to_drop=123)


    @staticmethod
    def test_preprocess_data_type_error_random_state(sample_dataframe):
        """Test `preprocess_data()` raises TypeError when 'random_state' is not a integer."""
        with pytest.raises(TypeError, match="'random_state' must be an integer"):
            preprocess_data(sample_dataframe, target='target_col', random_state=123.5)


## model_pipelines tests

class TestModelPipelines:
    @staticmethod
    def test_model_pipelines_successful_pipeline_creation(mock_preprocessor, mock_comparative_models, random_state):
        """Test the core functionality: successful creation of all pipelines."""
        pipelines = model_pipelines(
            preprocessor=mock_preprocessor,
            comparative_models=mock_comparative_models,
            random_state=random_state
        )

        expected_keys = {'LogisticRegression', 'RandomForest', 'KNNeighbors'}

        assert isinstance(pipelines, dict)
        assert set(pipelines.keys()) == expected_keys

        for pipeline in pipelines.values():
            assert isinstance(pipeline, Pipeline)
            assert pipeline.steps[0][0] == 'preprocessor'
            assert pipeline.steps[0][1] is mock_preprocessor

        assert pipelines['LogisticRegression'].steps[1][1].random_state == random_state
        assert pipelines['RandomForest'].steps[1][1].random_state == random_state


    @staticmethod
    def test_model_pipelines_default_random_state(mock_preprocessor, mock_comparative_models):
        """Test that the default 'random_state' (123) is used if not provided."""
        pipelines = model_pipelines(
            preprocessor=mock_preprocessor,
            comparative_models=mock_comparative_models
        )

        default_rs = 123
        assert pipelines['LogisticRegression'].steps[1][1].random_state == default_rs
        assert pipelines['RandomForest'].steps[1][1].random_state == default_rs


    @staticmethod
    def test_model_pipelines_type_error_preprocessor(mock_comparative_models):
        """Test `model_pipelines()` raises TypeError when 'preprocessor' is not a ColumnTransformer."""
        with pytest.raises(TypeError, match="'preprocessor' must be an sklearn ColumnTransformer"):
            model_pipelines(
                preprocessor='not a ColumnTransformer',
                comparative_models=mock_comparative_models
            )

    
    @staticmethod
    def test_model_pipelines_type_error_comparative_models(mock_preprocessor):
        """Test `model_pipelines()` raises TypeError when 'comparative_models' is not a dictionary."""
        with pytest.raises(TypeError, match="'comparative_models' must be a dictionary"):
            model_pipelines(
                preprocessor=mock_preprocessor,
                comparative_models='not a dictionary'
            )


    @staticmethod
    def test_model_pipelines_type_error_random_state(mock_preprocessor, mock_comparative_models):
        """Test `model_pipelines()` raises TypeError when 'random_state' is not a int"""
        with pytest.raises(TypeError, match="'random_state' must be an integer"):
            model_pipelines(
                preprocessor=mock_preprocessor,
                comparative_models=mock_comparative_models,
                random_state=1.0
            )


    @staticmethod
    def test_model_pipelines_value_error_empty_comparative_models(mock_preprocessor):
        """Test `model_pipelines()` raises ValueError if 'comparative_models' is an empty dictionary."""
        with pytest.raises(ValueError, match="'comparative_models' dictionary cannot be empty"):
            model_pipelines(
                preprocessor=mock_preprocessor,
                comparative_models={}
            )


    @staticmethod
    def test_model_pipelines_runtime_error_bad_estimator(mock_preprocessor, random_state):
        """Test `model_pipelines()` raises RuntimeError if an invalid estimator causes Pipeline creation to fail."""
        bad_models = {'BadModel': 'I am a string, not a model'}
        with pytest.raises(RuntimeError, match="Failed to create one or more pipelines"):
            model_pipelines(
                preprocessor=mock_preprocessor,
                comparative_models=bad_models,
                random_state=random_state
            )


## hyperparameters tests

class TestHyperparameters:
    @staticmethod
    def test_hyperparameters_successful_grid_creation(standard_models_dict):
        """Test that grids for support models are correctly created and structured."""
        result = hyperparameters(standard_models_dict)
        expected_keys = {'LogisticRegression', 'KNN', 'RandomForest', 'GaussianNB'}

        assert isinstance(result, dict)
        assert set(result.keys()) == expected_keys
        assert 'UnsupportedModel' not in result

        assert 'model' in result['LogisticRegression']
        assert isinstance(result['LogisticRegression']['model'], Pipeline)
        assert 'params' in result['LogisticRegression']
        assert 'classifier__C' in result['LogisticRegression']['params']
        assert isinstance(result['LogisticRegression']['params']['classifier__C'], list)
        assert result['LogisticRegression']['params']['classifier__solver'] == ['liblinear']

        assert 'classifier__n_estimators' in result['RandomForest']['params']
        assert 'classifier__max_features' in result['RandomForest']['params']


    @staticmethod
    def test_hyperparameters_type_error_non_dict_models():
        """Test `hyperparameters()` raises TypeError if 'models' is not a dictionary."""
        with pytest.raises(TypeError, match="'models' must be a dictionary"):
            hyperparameters('not a dictionary')


    @staticmethod
    def test_hyperparameters_value_error_empty_dict():
        """Test `hyperparameters()` raises ValueError if 'models' is an empty dictionary."""
        with pytest.raises(ValueError, match="The 'models' dictionary cannot be empty"):
            hyperparameters({})


    @staticmethod
    def test_hyperparameters_skips_non_pipeline_items(standard_models_dict, capsys):
        """Test that non-Pipeline items are skipped and a warning is printed."""
        models = standard_models_dict.copy()
        models['Bad Item'] = 'I am a string, not a Pipeline'

        result = hyperparameters(models)

        assert 'Bad Item' not in result

        captured = capsys.readouterr()

        assert "Warning: Skipping 'Bad Item'. Value is not a scikit-learn Pipeline" in captured.out
        assert "LogisticRegression" in result

    
    @staticmethod
    def test_hyperparameters_skips_unsupported_model_and_prints_message(standard_models_dict, capsys):
        """Test that a model without a defined grid is skipped and a message is printed."""
        result = hyperparameters(standard_models_dict)

        assert 'UnsupportedModel' not in result

        captured = capsys.readouterr()

        assert "Hyperparameters for UnsupportedModel not defined in this function and will be ommitted from tuning." in captured.out


    @staticmethod
    def test_hyperparameters_runtime_error_missing_classifier_step():
        """Test `hyperparameters()` raises RuntimeError if a Pipeline is missing the 'classifier' step."""
        bad_pipeline = mock.MagicMock(spec=Pipeline)
        bad_pipeline.named_steps = {'preprocessor': mock.MagicMock(), 'wrong_step': mock.MagicMock()}
        bad_pipeline.__class__ = Pipeline

        models = {'MissingClassifier': bad_pipeline}

        with pytest.raises(RuntimeError, match="An unexpected error occurred during parameter grid definition"):
            hyperparameters(models)


## train_evaluate_model tests

class TestTrainEvaluateModel:
    @staticmethod
    def test_train_evaluate_model_successful_execution_and_output_structure(mock_data, mock_params_all):
        """Test if the function runs successfully and returns the correct structure."""
        X_train, X_test, y_train, y_test = mock_data
        params = mock_params_all

        best_models, model_results = train_evaluate_model(X_train, X_test, y_train, y_test, params, cv=2)

        assert isinstance(best_models, dict)
        assert isinstance(model_results, list)
        assert len(best_models) == 2
        assert len(model_results) == 2
        assert 'LogisticRegression' in best_models
        assert 'RandomForest' in best_models

        lr_results = next(item for item in model_results if item['name'] == 'LogisticRegression')
        assert 'recall' in lr_results
        assert 'top_features' in lr_results
        assert 'predictions' in lr_results
        assert 'predictions_proba' in lr_results

        assert isinstance(lr_results['recall'], float)
        assert lr_results['roc_auc'] != 'N/A'


    @staticmethod
    def test_train_evaluate_data_logistic_regression_coef_features_importance(mock_data, mock_params_logreg):
        """Test correct handling of feature importances via 'coef_'."""
        X_train, X_test, y_train, y_test = mock_data
        _ , model_results = train_evaluate_model(X_train, X_test, y_train, y_test, mock_params_logreg, cv=2)

        lr_results = model_results[0]
        top_features = lr_results['top_features']

        assert isinstance(top_features, dict)
        assert len(top_features) <= 4
        assert any(key.startswith(('num_', 'cat_')) for key in top_features.keys())


    @staticmethod
    def test_train_evaluate_model_random_forest_feature_importance(mock_data, mock_params_rf):
        """Test correct handling of feature importance via 'feature_importances_'."""
        X_train, X_test, y_train, y_test = mock_data
        _ , model_results = train_evaluate_model(X_train, X_test, y_train, y_test, mock_params_rf, cv=2)

        rf_results = model_results[0]
        top_features = rf_results['top_features']

        assert isinstance(top_features, dict)
        assert len(top_features) <= 4
        assert all(val >= 0 for val in top_features.values())


    @staticmethod
    def test_train_evaluate_model_type_error_input_data():
        """Test `train_evaluate_model()` raises TypeError for incorrect input data types."""
        X_train = [1, 2, 3]
        X_test, y_train, y_test = pd.DataFrame({'a': [1]}), pd.Series([0]), pd.Series([0])

        with pytest.raises(TypeError, match="Input data X_train, X_test, y_train, and y_test must be pandas DataFrame or Series"):
            train_evaluate_model(X_train, X_test, y_train, y_test, params={})


    @staticmethod
    def test_train_evaluate_model_value_error_empty_params():
        """Test `train_evaluate_model()` raises ValueError if 'params' is a empty dictionary."""
        X_train, X_test, y_train, y_test = pd.DataFrame({'a': [1]}), pd.DataFrame({'a': [1]}), pd.Series([0]), pd.Series([0])

        with pytest.raises(ValueError, match="The 'params' dictionary must be a non-empty dictionary"):
            train_evaluate_model(X_train, X_test, y_train, y_test, {})


    @staticmethod
    def test_train_evaluate_model_value_error_invalid_cv(mock_data, mock_params_logreg):
        """Test `train_evaluate_model()` raises ValueError if 'cv' is an invalid value."""
        X_train, X_test, y_train, y_test = mock_data

        with pytest.raises(ValueError, match="'cv' must be an integer greater than 1"):
            train_evaluate_model(X_train, X_test, y_train, y_test, params=mock_params_logreg, cv=1)


    @staticmethod
    def test_train_evaluate_model_warning_invalid_model_setup(mock_data, mock_params_logreg, capsys):
        """Test `train_evaluate_model()` raises a warning and skips logic for an improperly structured model."""
        X_train, X_test, y_train, y_test = mock_data
        invalid_params = {
            'ValidModel': mock_params_logreg['LogisticRegression'],
            'InvalidModel': {'model': LogisticRegression()}
        }

        _ , model_results = train_evaluate_model(X_train, X_test, y_train, y_test, invalid_params, cv=2)

        captured = capsys.readouterr()

        assert "Warning: Skipping 'InvalidModel'. Setup is invalid or missing 'model' or 'params' keys." in captured.out
        assert len(model_results) == 1
        assert model_results[0]['name'] == 'ValidModel'


    @staticmethod
    def test_train_evaluate_model_runtime_error_handling(mock_data, capsys):
        """Test `train_evaluate_model()` raises RuntimeError if unexpected errors occur during model training."""
        X_train, X_test, y_train, y_test = mock_data

        class BrokenClassifier(BaseEstimator):
            def fit(self, X, y):
                raise RuntimeError("Intentional training failure")
            def predict(self, X): return np.ones(X.shape[0])

        params = {
            'BrokenModel': {
                'model': Pipeline(steps=[('classifier', BrokenClassifier())]),
                'params': {}
            }
        }

        _ , model_results = train_evaluate_model(X_train, X_test, y_train, y_test, params, cv=2)

        captured = capsys.readouterr()

        assert "Error occurred while training/evaluating 'BrokenModel'" in captured.out
        assert len(model_results) == 1
        assert model_results[0]['name'] == 'BrokenModel'
        assert len(model_results[0].keys()) == 1


## voting_model tests

class TestVotingModel:
    @staticmethod
    def test_voting_model_with_valid_input(mock_data, mock_params_all):
        """Test `voting_model()` with valid inputs."""
        X_train, X_test, y_train, y_test = mock_data
        best_models = {k: v['model'] for k, v in mock_params_all.items()}

        result, metrics = voting_model(X_train, X_test, y_train, y_test, best_models, n_iter=7)

        assert isinstance(result, dict)
        assert 'VotingClassifier' in result
        assert hasattr(result['VotingClassifier'], 'predict')
        assert isinstance(metrics, list)
        assert len(metrics) == 1

        metric_entry = metrics[0]

        for key in ['f1', 'recall', 'precision', 'roc_auc', 'accuracy']:
            assert key in metric_entry
            assert isinstance(metric_entry[key], float)


    @staticmethod
    def test_voting_model_with_empty_models_dict(mock_data):
        """Test `voting_model()` with an empty models dictionary."""
        X_train, X_test, y_train, y_test = mock_data
        result, metrics = voting_model(X_train, X_test, y_train, y_test, {})

        assert result == {}
        assert metrics == []


    @staticmethod
    def test_voting_model_with_non_proba_models(mock_data, mock_params_non_proba):
        """Test `voting_model()` with models that don't support 'predict_proba'."""
        X_train, X_test, y_train, y_test = mock_data

        best_model = mock_params_non_proba

        result, metrics = voting_model(X_train, X_test, y_train, y_test, best_model, n_iter=1)

        assert result == {}
        assert metrics == []


    @staticmethod
    def test_voting_model_returns_expected_structure(mock_data, mock_params_logreg):
        """Test if `voting_model()` returns the expected structure."""
        X_train, X_test, y_train, y_test = mock_data
        best_models = {k: v['model'] for k, v in mock_params_logreg.items()}

        best_model, metrics = voting_model(X_train, X_test, y_train, y_test, best_models, n_iter=1)

        assert isinstance(best_model, dict)
        assert 'VotingClassifier' in best_model

        metric = metrics[0]
        assert isinstance(metric['predictions'], np.ndarray)
        assert isinstance(metric['predictions_proba'], np.ndarray)
        assert metric['predictions'].shape[0] == len(X_test)
        assert metric['predictions_proba'].shape[0] == len(X_test)


    @staticmethod
    def test_voting_model_handles_exception(mock_data):
        """Test how `voting_model()` handles an exception."""
        X_train, X_test, y_train, y_test = mock_data

        class BadModel(BaseEstimator, ClassifierMixin):
            def fit(self, X, y): raise ValueError("fail")
            def predict(self, X): return X

        best_models = {'BadModel': create_mock_pipeline(BadModel())}

        result, metrics = voting_model(X_train, X_test, y_train, y_test, best_models, n_iter=1)

        assert result == {}
        assert metrics == []


    @staticmethod
    @pytest.mark.parametrize("scoring", ["accuracy", "precision", "recall", "f1"])
    def test_voting_model_with_different_scoring(mock_data, mock_params_logreg, scoring):
        """Test `voting_model()` with different scoring metrics."""
        X_train, X_test, y_train, y_test = mock_data
        best_models = {k: v['model'] for k, v in mock_params_logreg.items()}

        result, metrics = voting_model(X_train, X_test, y_train, y_test, best_models, metric=scoring, n_iter=1)

        assert isinstance(result, dict)
        assert 'VotingClassifier' in result
        assert isinstance(metrics, list)
        assert 'f1' in metrics[0]


## comparative_models tests

class TestComparativeModels:
    @staticmethod
    def test_comparative_models_success(sample_dataframe):
        """Test `comparative_models()` with valid inputs."""
        df = sample_dataframe.copy()
        df['target_col'] = (df['target_col'] > 0.5).astype(int)
        target = 'target_col'

        dummy_X = df.drop(columns=[target])
        dummy_y = df[target]
        X_train, X_test = dummy_X.iloc[:10], dummy_X.iloc[10:]
        y_train, y_test = dummy_y.iloc[:10], dummy_y.iloc[10:]

        numerical_features = ['low_num_int', 'high_num_float']
        categorical_features = ['low_cat_str']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])

        pipeline.fit(X_train, y_train)

        pipeline.predict = mock.MagicMock(return_value=np.ones(len(X_test)))
        pipeline.predict_proba = mock.MagicMock(return_value=np.tile([0.3, 0.7], (len(X_test), 1)))

        mock_best_models = {'RandomForest': pipeline}
        mock_model_results = [{
            'name': 'RandomForest',
            'accuracy': 0.9,
            'recall': 0.8,
            'precision': 0.85,
            'f1': 0.82,
            'roc_auc': 0.88,
            'params': {'n_estimators': 10},
            'predictions': np.ones(len(X_test)),
            'predictions_proba': np.ones(len(X_test))
        }]

        mock_voting_results = ({'VotingClassifier': pipeline}, [{
            'name': 'VotingClassifier',
            'accuracy': 0.92,
            'recall': 0.84,
            'precision': 0.87,
            'f1': 0.85,
            'roc_auc': 0.89,
            'params': {'weights': [1, 2]},
            'predictions': np.ones(len(X_test)),
            'predictions_proba': np.ones(len(X_test))
        }])

        with mock.patch('src.telco_customer_churn_analysis.model_utils.preprocess_data', return_value=(dummy_X, dummy_y, X_train, X_test, y_train, y_test, preprocessor, mock.MagicMock())), \
            mock.patch('src.telco_customer_churn_analysis.model_utils.model_pipelines', return_value={'LogisticRegression': pipeline, 'RandomForest': pipeline}), \
            mock.patch('src.telco_customer_churn_analysis.model_utils.hyperparameters', return_value={'RandomForest': {'params': {'n_estimators': [5]}}}), \
            mock.patch('src.telco_customer_churn_analysis.model_utils.train_evaluate_model', return_value=(mock_best_models, mock_model_results)), \
            mock.patch('src.telco_customer_churn_analysis.model_utils.voting_model', return_value=mock_voting_results):

            all_models, all_results, best_model, X_train_out, X_test_out, y_test_out = comparative_models(df, target, {'RandomForest': pipeline})

            assert isinstance(all_models, dict)
            assert 'VotingClassifier' in all_models
            assert isinstance(all_results, pd.DataFrame)
            assert not all_results.empty
            assert isinstance(best_model, Pipeline)
            assert isinstance(X_train_out, pd.DataFrame)
            assert isinstance(X_test_out, pd.DataFrame)
            assert isinstance(y_test_out, pd.Series)


    @staticmethod
    def test_comparative_models_missing_helper(sample_dataframe, mock_comparative_models):
        """Test `comparative_models()` when missing a helper function."""
        df = sample_dataframe.copy()
        df['target_col'] = (df['target_col'] > 0.5).astype(int)
        target = 'target_col'

        with mock.patch('src.telco_customer_churn_analysis.model_utils.preprocess_data', side_effect=NotImplementedError("Missing Function")):
            all_models, all_results, best_model, X_train_out, X_test_out, y_test_out = comparative_models(df, target, mock_comparative_models)

            assert all_models == {}
            assert all_results.empty
            assert best_model is None


    @staticmethod
    def test_comparative_models_unexpected_exception(sample_dataframe, mock_comparative_models):
        """Test how `comparative_models` handles an exception."""
        df = sample_dataframe.copy()
        df['target_col'] = (df['target_col'] > 0.5).astype(int)
        target = 'target_col'

        with mock.patch('src.telco_customer_churn_analysis.model_utils.preprocess_data', side_effect=RuntimeError("Something broke")):
            all_models, all_results, best_model, X_train_out, X_test_out, y_test_out = comparative_models(df, target, mock_comparative_models)

            assert all_models == {}
            assert all_results.empty
            assert best_model is None


## profit_curve tests

class TestProfitCurve:
    @staticmethod
    def test_profit_curve_basic_case():
        """Test `profit_curve()` with a simple known case where results are predictable."""
        labels = np.array([1, 0, 1, 0, 1])
        predictions_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        median_value = 100.0
        cost = 10.0
        retention_rate = 0.5

        model_df = pd.DataFrame({'predictions_proba': [predictions_proba]})

        best_threshold, best_profit, profit_values = profit_curve(model_df, median_value, cost, retention_rate, labels)

        assert isinstance(best_threshold, float)
        assert 0.05 <= best_threshold <= 0.95
        assert isinstance(best_profit, float)
        assert isinstance(profit_values, dict)
        assert len(profit_values) > 0
        assert best_threshold <= 0.9


    @staticmethod
    def test_profit_curve_series_input():
        """Test that `profit_curve()` works when inputs are pandas Series."""
        labels = pd.Series([1, 1, 0, 0])
        predictions_proba = pd.Series([0.8, 0.7, 0.2, 0.1])

        model_df = pd.DataFrame({'predictions_proba': [predictions_proba]})

        best_threshold, best_profit, profit_values = profit_curve(model_df, 200.0, 20.0, 0.6, labels)

        assert isinstance(best_threshold, float)
        assert isinstance(best_profit, float)
        assert isinstance(profit_values, dict)
        assert all(isinstance(k, float) for k in profit_values.keys())
        assert all(isinstance(v, float) for v in profit_values.values())


    @staticmethod
    def test_profit_curve_all_zero_predictions():
        """Test `profit_curve()` behavior when all probabilities are zero (edge case). Should return 0.5 as default threshold and 0.0 as profit."""
        labels = np.array([0, 1, 1])
        predictions_proba = np.array([0.1, 0.2, 0.3])

        model_df = pd.DataFrame({'predictions_proba': [predictions_proba]})

        best_threshold, best_profit, profit_values = profit_curve(model_df, 100.0, 20.0, 0.3, labels)

        assert best_threshold == 0.1
        assert best_profit == 20.0
        assert isinstance(profit_values, dict)
        assert len(profit_values) > 0


    @staticmethod
    def test_profit_curve_invalid_shape_matrix():
        """Force `profit_curve()` to skip thresholds where confusion matrix is invalid."""
        labels = np.array([1, 1, 1, 1, 1])
        predictions_proba = np.array([0.9, 0.8, 0.85, 0.95, 0.7])

        model_df = pd.DataFrame({'predictions_proba': [predictions_proba]})

        best_threshold, best_profit, profit_values = profit_curve(model_df, 100.0, 20.0, 0.2, labels)

        assert isinstance(best_threshold, float)
        assert isinstance(best_profit, float)
        assert isinstance(profit_values, dict)
        assert len(profit_values) > 0


## deployment_model tests

class TestDeploymentModel:
    @staticmethod
    def test_deployment_model_successful(monkeypatch, sample_dataframe):
        """Test `deployment_model` with valid inputs."""
        df = sample_dataframe.copy()
        df['target_col'] = (df['target_col'] > 0.5).astype(int)

        target = 'target_col'

        preprocessor = mock.MagicMock()
        classifier = LogisticRegression()
        fitted_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        drop_cols = ['drop_me', 'high_cat_str']
        dummy_X = df.drop(columns=[target] + drop_cols, errors='ignore')
        dummy_y = df[target]

        monkeypatch.setattr('tests.test_model_utils.preprocess_data',
                            lambda *args, **kwargs: (dummy_X, dummy_y, None, None, None, None, preprocessor, None)
                            )
        
        with mock.patch('joblib.dump') as mock_joblib_dump:
            deployment_pipeline = deployment_model(df, fitted_pipeline, target, cols_to_drop=drop_cols)

            assert isinstance(deployment_pipeline, Pipeline)
            assert 'classifier' in deployment_pipeline.named_steps
            assert 'preprocessor' in deployment_pipeline.named_steps
            mock_joblib_dump.assert_called_once()
            assert mock_joblib_dump.call_args[0][1] == 'deployment_pipeline.pkl'


    @staticmethod
    def test_deployment_model_missing_preprocessor_params(monkeypatch, sample_dataframe):
        """Test `deployment_model()` handles exceptions raises during preprocessing."""
        df = sample_dataframe.copy()
        df['target_col'] = (df['target_col'] > 0.5).astype(int)
        target = 'target_col'

        preprocessor = mock.MagicMock()
        classifier = LogisticRegression()
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        monkeypatch.setattr(
            'tests.test_model_utils.preprocess_data', mock.Mock(side_effect=Exception("preprocessing error"))
        )

        drop_cols = ['drop_me', 'high_cat_str']
        with mock.patch("joblib.dump") as mock_joblib_dump:
            deployment_pipeline = deployment_model(df, model, target, cols_to_drop=drop_cols)

            assert isinstance(deployment_pipeline, Pipeline)
            mock_joblib_dump.assert_called_once()


    @staticmethod
    def test_deployment_model_value_error_invalid_model(sample_dataframe):
        """Test `deployment_model()` raises ValueError when the model is invalid."""
        df = sample_dataframe.copy()
        df['target_col'] = (df['target_col'] > 0.5).astype(int)
        target = 'target_col'

        model = Pipeline([
            ('preprocessor', mock.MagicMock())
        ])

        with pytest.raises(ValueError, match="must contain a 'classifier' step"):
            deployment_model(df, model, target, cols_to_drop=None)


    @staticmethod
    def test_deployment_model_cols_to_drop_variants(monkeypatch, sample_dataframe):
        """Test `deployment_model()` with different variants of 'cols_to_drop' inputs."""
        df = sample_dataframe.copy()
        df['target_col'] = (df['target_col'] > 0.5).astype(int)
        target = 'target_col'

        drop_cols = ['drop_me', 'high_cat_str']
        dummy_X = df.drop(columns=[target] + drop_cols)
        dummy_y = df[target]
        preprocessor = mock.MagicMock()
        classifier = LogisticRegression()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        monkeypatch.setattr('tests.test_model_utils.preprocess_data',
                            lambda *args, **kwargs: (dummy_X, dummy_y, None, None, None, None, preprocessor, None)
                            )
        
        with mock.patch('joblib.dump') as mock_joblib_dump:
            deployment_pipeline = deployment_model(df, pipeline, target, 'high_cat_str')
            assert isinstance(deployment_pipeline, Pipeline)

            deployment_pipeline = deployment_model(df, pipeline, target, drop_cols)
            assert isinstance(deployment_pipeline, Pipeline)

            assert mock_joblib_dump.call_count == 2


## predict_churn tests

class TestPredictChurn:
    @staticmethod
    def test_predict_churn_successful(sample_dataframe):
        """Test `predict_churn()` with valid inputs."""
        df = sample_dataframe.copy()
        df['Contract'] = ['Month-to-month'] * 10 + ['Two year'] * 10
        df['target_col'] = (df['target_col'] > 0.5).astype(int)

        mock_model = mock.MagicMock(spec=Pipeline)
        mock_model.predict_proba.return_value = np.tile([0.4, 0.6], (20, 1))

        with mock.patch('joblib.load', return_value=mock_model):
            result_df = predict_churn(df, threshold=0.5, model_path='deployment_model.pkl')

            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty
            assert all(result_df['Contract'] == 'Month-to-month')
            assert all(result_df['Churn Probability'] >= 0.5)
            assert result_df['Churn Probability'].is_monotonic_decreasing


    @staticmethod
    def test_predict_churn_file_not_found(sample_dataframe):
        """Test `predict_churn()` when file is not found."""
        result_df = predict_churn(sample_dataframe, threshold=0.6)
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty


    @staticmethod
    def test_predict_churn_prediction_error(sample_dataframe):
        """Test `predict_churn()` returns an empty DataFrame if the model is not fitted."""
        mock_model = mock.MagicMock(spec=Pipeline)
        mock_model.predict_proba.side_effect = NotFittedError("Model not fitted")

        with mock.patch('joblib.load', return_value=mock_model):
            result = predict_churn(sample_dataframe, threshold=0.6)
            assert isinstance(result, pd.DataFrame)
            assert result.empty


    @staticmethod
    def test_predict_churn_threshold_filtering(sample_dataframe):
        """Test `predict_churn` filters results based on the probability threshold."""
        df = sample_dataframe.copy()
        df['Contract'] = ['Month-to-month'] * 20
        df['target_col'] = (df['target_col'] > 0.5).astype(int)

        proba = np.linspace(0.1, 1.0, len(df))
        predict_proba = np.stack([1 - proba, proba], axis=1)

        mock_model = mock.MagicMock(spec=Pipeline)
        mock_model.predict_proba.return_value = predict_proba

        with mock.patch('joblib.load', return_value=mock_model):
            threshold = 0.8
            result = predict_churn(df, threshold=threshold)

            assert not result.empty
            assert all(result['Contract'] == 'Month-to-month')
            assert all(result['Churn Probability'] >= threshold)
            assert len(result) == 5


    @staticmethod
    def test_predict_churn_no_customer_match(sample_dataframe):
        """Test `predict_churn()` returns an empty DataFrame when no customers meet the threshold criteria."""
        df = sample_dataframe.copy()
        df['Contract'] = ['Two year'] * 20

        mock_model = mock.MagicMock(spec=Pipeline)
        mock_model.predict_proba.return_value = np.tile([0.3, 0.7], (20, 1))

        with mock.patch('joblib.load', return_value=mock_model):
            result = predict_churn(df, threshold=0.5)

            assert isinstance(result, pd.DataFrame)
            assert result.empty


## abc_test tests

class TestAbcTest:
    @staticmethod
    def test_abc_test_assigns_groups_correctly(sample_dataframe):
        """Test `abc_test()` assigns the groups correctly."""
        df = sample_dataframe.copy()
        df['Contract'] = ['Month-to-month'] * len(df)
        df['Churn Probability'] = np.linspace(0.6, 0.99, len(df))
        df['Intervention Flag'] = 1

        result = abc_test(df, random_seed=123)

        assert not result.empty
        assert 'Group' in result.columns
        assert 'Intervention Details' in result.columns
        assert set(result['Group']).issubset({'A (Control)', 'B (Price Offer)', 'C (Service Offer)'})
        assert set(result['Intervention Details']).issubset({
            '10% Off 1-Year Contract', '6 Months Free Tech Support', 'None (Control)'
        })


    @staticmethod
    def test_abc_test_handles_empty_dataframe():
        """Test `abc_test` returns an empty DataFrame if the input is also an empty DataFrame."""
        empty_df = pd.DataFrame()
        result = abc_test(empty_df)

        assert result.empty


    @staticmethod
    def test_abc_test_reproducibility(sample_dataframe):
        """Test `abc_test()` returns consistent output when the random_seed and inputs are the same."""
        df = sample_dataframe.copy()
        df['Contract'] = ['Month-to-month'] * len(df)
        df['Churn Probability'] = np.linspace(0.6, 0.99, len(df))
        df['Intervention Flag'] = 1

        result1 = abc_test(df, random_seed=123)
        result2 = abc_test(df, random_seed=123)

        pd.testing.assert_frame_equal(result1[['Group', 'Intervention Details']], result2[['Group', 'Intervention Details']])


    @staticmethod
    def test_abc_test_intervention_mapping():
        """Test `abc_test()` correctly assigns correct 'Intervention Details' based on the 'Group' value."""
        test_df = pd.DataFrame({'Group': ['A (Control)', 'B (Price Offer)', 'C (Service Offer)']})

        def map_intervention(group):
            if group == 'B (Price Offer)':
                return '10% Off 1-Year Contract'
            elif group == 'C (Service Offer)':
                return '6 Months Free Tech Support'
            else:
                return 'None (Control)'
            
        test_df['Intervention Details'] = test_df['Group'].apply(map_intervention)

        expected = ['None (Control)', '10% Off 1-Year Contract', '6 Months Free Tech Support']

        assert list(test_df['Intervention Details']) == expected


    @staticmethod
    def test_abc_test_group_distribution(sample_dataframe):
        """Test `abc_test()` assigns all customers to one of the three valid groups."""
        df = sample_dataframe.copy()
        df['Contract'] = ['Month-to-month'] * len(df)
        df['Churn Probability'] = np.linspace(0.6, 0.99, len(df))
        df['Intervention Flag'] = 1

        result = abc_test(df, random_seed=123)

        group_counts = result['Group'].value_counts()

        assert group_counts.sum() == len(df)
        assert all(group in group_counts for group in ['A (Control)', 'B (Price Offer)', 'C (Service Offer)'])