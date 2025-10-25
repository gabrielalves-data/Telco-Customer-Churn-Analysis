import pytest
import pandas as pd
import numpy as np
from unittest import mock
import os
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.telco_customer_churn_analysis.telco_customer_churn_analysis import (data_preprocessing, exploratory_analysis, bin_df,
                                                                             hypothesis_test, get_model, global_explainer,
                                                                             local_explainer, profit_curve_threshold, deploy_model,
                                                                             generate_test_data, predict_df, abc_test_assignment)

@pytest.fixture(autouse=True)
def suppress_show(monkeypatch):
    """Automatically suppress matplotlib's plt.show() during tests to prevent plots from displaying."""

    monkeypatch.setattr(plt, 'show', lambda: None)


@pytest.fixture(autouse=True)
def close_figures_after_test():
    yield
    plt.close('all')

    
@pytest.fixture
def mock_dataframe():
    """Return a small mock DataFrame simulating Telco customer data for testing purposes."""

    data = {
        'Total Charges': ['20', 'nan', '40'],
        'Monthly Charges': [10, 15, 20],
        'Tenure Months': [2, 3, 2],
        'Churn Value': [0, 1, 0],
        'CustomerID': [1, 2, 3],
        'Count': [1, 1, 1],
        'Country': ['US', 'US', 'US'],
        'Lat Long': ['0,0', '0,0', '0,0'],
        'Churn Label': ['No', 'Yes', 'No'],
        'City': ['San Jose', 'San Diego', 'San Francisco']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df():
    """Return a sample DataFrame with numeric and categorical features for testing pipelines and transformations."""

    data = {
        'City': ['A', 'A', 'B', 'C', 'C', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
        'Gender': ['Male', 'Female'] * 7,
        'Senior Citizen': [0, 1] * 7,
        'Partner': ['Yes', 'No'] * 7,
        'Tenure Months': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135],
        'Phone Service': ['Yes'] * 14,
        'Multiple Lines': ['No'] * 14,
        'Internet Service': ['DSL'] * 14,
        'Online Security': ['No'] * 14,
        'Online Backup': ['Yes'] * 14,
        'Device Protection': ['No'] * 14,
        'Tech Support': ['Yes'] * 14,
        'Dependents': ['No'] * 14,
        'Streaming TV': ['No'] * 14,
        'Streaming Movies': ['Yes'] * 14,
        'Contract': ['Month-to-month'] * 14,
        'Paperless Billing': ['Yes'] * 14,
        'Payment Method': ['Electronic check'] * 14,
        'Monthly Charges': np.linspace(20, 100, 14),
        'Total Charges': np.linspace(100, 1400, 14),
        'Churn Value': [0,1]*7,
        'Churn Score': np.linspace(0, 100, 14),
        'CLTV': np.linspace(1000, 6000, 14),
        'Churn Reason': ['Reason1', 'Reason2'] * 7,
    }

    return pd.DataFrame(data)


def fake_bin_and_plot(title, label, df, col, new_col, bins, labels=None, show_plot=True, ax=None, **kwargs):
    """Fake implementation of bin_and_plot for testing. Randomly assigns categorical values to a new column."""

    df = df.copy()
    n = len(df)

    if labels is not None:
        df[new_col] = pd.Categorical(np.random.choice(labels, n), categories=labels, ordered=True)
    else:
        df[new_col] = pd.Categorical([], categories=labels, ordered=True)
    return df
    

@pytest.fixture(autouse=True)
def patch_bin_and_plot(monkeypatch):
    """Patch the bin_and_plot function in the module to use fake_bin_and_plot during tests."""

    monkeypatch.setattr("src.telco_customer_churn_analysis.telco_customer_churn_analysis.bin_and_plot", fake_bin_and_plot)
    yield


@pytest.fixture
def mock_comparative_models_return():
    """Return mock objects representing model training results and test data for comparative model testing."""

    fake_model_results = {'KNeighbors': 'model1', 'RandomForest': 'model2'}
    fake_best_model = 'best_model'
    fake_X_train = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    fake_X_test = pd.DataFrame({'feature1': [5], 'feature2': [6]})
    fake_y_test = pd.Series([0])

    return {}, fake_model_results, fake_best_model, fake_X_train, fake_X_test, fake_y_test


@pytest.fixture
def simple_pipeline(sample_df):
    """Return a simple RandomForest pipeline fitted on the sample_df along with feature DataFrame."""

    numeric_features = ['Monthly Charges', 'Total Charges', 'Tenure Months']
    categorical_features = ['City', 'Gender']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    clf = RandomForestClassifier(random_state=123)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    X = sample_df[numeric_features + categorical_features]
    y = sample_df['Churn Value']
    pipe.fit(X, y)
    
    return pipe, X


@pytest.fixture
def mock_prediction_df():
    """Return a small DataFrame representing new data for predictions."""

    return pd.DataFrame({
        'Monthly Charges': [25.0, 75.0],
        'Total Charges': [200.0, 500.0],
        'Tenure Months': [10, 20],
        'Contract': ['Month-to-month', 'Two year']
    })


@pytest.fixture
def mock_model():
    """Mock a scikit-learn pipeline with predict_proba()."""
    mock_pipe = mock.MagicMock()
    mock_pipe.predict_proba.return_value = np.array([
        [0.2, 0.8],
        [0.9, 0.1]
    ])

    return mock_pipe


## data_preprocessing tests

class TestDataPreprocessing:
    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.kaggle_download')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.read_excel')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_head')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.col_replace')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.null_rows')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_loc')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_aggfunc')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.drop_labels')
    def test_data_preprocessing(
        mock_drop_labels,
        mock_df_aggfunc,
        mock_df_loc,
        mock_null_rows,
        mock_col_replace,
        mock_df_head,
        mock_read_excel,
        mock_kaggle_download,
        mock_dataframe,
        capsys
    ):
        """Test data preprocessing and verify output columns and printed stats."""

        mock_kaggle_download.return_value = '/fake/path/Telco_customer_churn.xlsx'
        mock_read_excel.return_value = mock_dataframe

        mock_df_head.return_value = None
        mock_col_replace.side_effect = lambda df, col, old, new: df
        mock_null_rows.side_effect = lambda df, col=None: pd.Series([False, True, False], index=df.index)
        def df_loc_side_effect(df, mask, col):
            return df.loc[mask, col]
        mock_df_loc.side_effect = df_loc_side_effect

        mock_df_aggfunc.side_effect = lambda df, func, col: pd.Series([0])

        def drop_labels_side_effect(df, cols):
            return df.drop(cols, axis=1)
        mock_drop_labels.side_effect = drop_labels_side_effect

        import src.telco_customer_churn_analysis.telco_customer_churn_analysis
        result_df = src.telco_customer_churn_analysis.telco_customer_churn_analysis.data_preprocessing()

        mock_kaggle_download.assert_called_once()
        mock_read_excel.assert_called_once_with('/fake/path/Telco_customer_churn.xlsx')
        mock_df_head.assert_called_once()
        assert 'Total Charges' in result_df.columns
        assert 'CustomerID' not in result_df.columns

        captured = capsys.readouterr()
        assert 'Number of Missing Values' in captured.out
        assert 'Number of Rows by Churn Value' in captured.out
        assert 'Mean of Churn Value' in captured.out
        assert 'Median of Monthly Charges' in captured.out


## exploratory_analysis tests

class TestExploratoryAnalysis:
    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.plt.show')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.bin_and_plot')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.heatmap')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_aggfunc')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.histogram')
    @mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.count_plot')
    def test_exploratory_analysis(
        mock_count_plot,
        mock_histogram,
        mock_df_aggfunc,
        mock_heatmap,
        mock_bin_and_plot,
        mock_plt_show,
        sample_df,
        capsys
    ):
        """Test exploratory analysis calls plotting functions and adds binned columns."""

        mock_count_plot.return_value = None
        mock_histogram.return_value = None
        mock_heatmap.return_value = None
        mock_df_aggfunc.return_value = sample_df['Churn Reason'].value_counts()

        mock_bin_and_plot.side_effect = fake_bin_and_plot

        result_df = exploratory_analysis(sample_df)

        assert mock_count_plot.called
        assert mock_histogram.called
        assert mock_bin_and_plot.called
        assert mock_heatmap.called
        assert mock_df_aggfunc.called
        mock_plt_show.assert_called()

        assert 'Tenure Group' in result_df.columns
        assert 'Churn Probability' in result_df.columns
        assert 'Customer Value' in result_df.columns

        mock_histogram.assert_any_call('Distribution of Churned Customers Monthly Charges', 'Monthly Charges', mock.ANY, 'Monthly Charges', 20, ax=mock.ANY)
        mock_histogram.assert_any_call('Distribution of Churned Customers Tenure Months', 'Tenure Months', mock.ANY, 'Tenure Months', 20, ax=mock.ANY)
        mock_histogram.assert_any_call('Distribution of Churned Customers Churn Score', 'Churn Score', mock.ANY, 'Churn Score', 20, ax=mock.ANY)
        mock_histogram.assert_any_call('Distribution of Churned Customers CLTV', 'CLTV', mock.ANY, 'CLTV', 20, ax=mock.ANY)


## bin_df tests

class TestBinDf:
    @staticmethod
    def test_bin_df_returns_dataframe_with_bins(sample_df):
        """Test bin_df creates binned categorical columns with correct categories."""

        df_binned = bin_df(sample_df, show=False)
        
        assert 'Tenure Group' in df_binned.columns
        assert 'Churn Probability' in df_binned.columns
        assert 'Customer Value' in df_binned.columns

        tenure_cats = ['New Customer', 'New/Established Customer', 'Established/Veteran Customer', 'Veteran Customer']
        churn_cats = ['Less Probability of Churn', 'Less/Moderate Probability of Churn', 'Moderate/High Probability of Churn', 'High Probability of Churn']
        value_cats = ['Low Value', 'Low/Mid Value', 'Mid/High Value', 'High Value']

        assert isinstance(df_binned['Tenure Group'].dtype, pd.CategoricalDtype)
        assert list(df_binned['Tenure Group'].cat.categories) == tenure_cats

        assert isinstance(df_binned['Churn Probability'].dtype, pd.CategoricalDtype)
        assert list(df_binned['Churn Probability'].cat.categories) == churn_cats

        assert isinstance(df_binned['Customer Value'].dtype, pd.CategoricalDtype)
        assert list(df_binned['Customer Value'].cat.categories) == value_cats


    @staticmethod
    def test_bin_df_raises_keyerror_for_missing_columns():
        """Test bin_df raises KeyError if required columns are missing."""

        df_missing = pd.DataFrame({'A': [1, 2, 3]})
        
        with pytest.raises(KeyError):
            bin_df(df_missing, show=False)


    @staticmethod
    def test_bin_df_handles_empty_dataframe():
        """Test bin_df handles empty dataframe and still creates columns."""

        empty_df = pd.DataFrame({
            'Tenure Months': pd.Series([], dtype='int64'),
            'Churn Score': pd.Series([], dtype='int64'),
            'CLTV': pd.Series([], dtype='int64')
        })

        result = bin_df(empty_df, show=False)
        
        assert 'Tenure Group' in result.columns
        assert 'Churn Probability' in result.columns
        assert 'Customer Value' in result.columns
        assert result.empty


    @staticmethod
    def test_bin_df_plot_axes_layout(monkeypatch, sample_df):
        """Test bin_df plots correctly without displaying them."""

        import matplotlib
        matplotlib.use('Agg')
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

        df_result = bin_df(sample_df, show=True)

        assert isinstance(df_result, pd.DataFrame)


## hypothesis_test tests

class TestHypothesisTest:
    @staticmethod
    def test_hypothesis_test_with_test_data(mock_dataframe):
        """Test `hypothesis_test()` using 'Test' data with specified columns."""

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.data_preprocessing', return_value=mock_dataframe), \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.bin_df', return_value=mock_dataframe) as mock_bin, \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.chi_squared_test', return_value=(1.0, 0.05, True)) as mock_chi:

            hypothesis_test(data_choice='Test', col1='Monthly Charges', col2='Churn Value')

            mock_bin.assert_called_once_with(mock_dataframe)
            mock_chi.assert_called_once_with(mock_dataframe, 'Monthly Charges', 'Churn Value')


    @staticmethod
    def test_hypothesis_test_with_new_data(sample_df):
        """Test `hypothesis_test()` using 'New' synthetic data with specified columns."""

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.generate_test_data', return_value=sample_df), \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.bin_df_2', return_value=sample_df) as mock_bin2, \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.chi_squared_test', return_value=(1.0, 0.05, True)) as mock_chi:

            hypothesis_test(data_choice='New', col1='City', col2='Gender')

            mock_bin2.assert_called_once_with(sample_df)
            mock_chi.assert_called_once_with(sample_df, 'City', 'Gender')


    @staticmethod
    def test_hypothesis_test_invalid_choice(mock_dataframe):
        """Ensure `hypothesis_test()` raises ValueError for invalid data_choice input."""

        with pytest.raises(ValueError, match='Input invalid'):
            hypothesis_test(data_choice='InvalidChoice')


    @staticmethod
    def test_hypothesis_test_default_columns(mock_dataframe):
        """Test that `hypothesis_test()` uses default first two columns when col1 and col2 are not provided."""

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.data_preprocessing', return_value=mock_dataframe), \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.bin_df', return_value=mock_dataframe), \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.chi_squared_test', return_value=(1.0, 0.05, True)) as mock_chi:

            hypothesis_test(data_choice='Test')

            mock_chi.assert_called_once_with(mock_dataframe, 'Total Charges', 'Monthly Charges')


    @staticmethod
    def test_hypothesis_test_insufficient_columns():
        """Ensure `hypothesis_test()` raises ValueError when DataFrame has less than 2 columns."""

        df_one_col = pd.DataFrame({'OnlyCol': [1,2,3]})
        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.data_preprocessing', return_value=df_one_col), \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.bin_df', return_value=df_one_col):
            with pytest.raises(ValueError, match='Please select an dataframe column.'):
                hypothesis_test(data_choice='Test')


## get_model tests

class TestGetModel:
    @staticmethod
    def test_get_model_returns_expected(monkeypatch, sample_df, mock_comparative_models_return):
        """Test get_model returns expected outputs from comparative_models."""

        monkeypatch.setattr("src.telco_customer_churn_analysis.telco_customer_churn_analysis.comparative_models", lambda *args, **kwargs: mock_comparative_models_return)

        all_models, all_results, best_model, X_train, X_test, y_test = get_model(sample_df)

        assert all_results == mock_comparative_models_return[1]
        assert best_model == mock_comparative_models_return[2]
        pd.testing.assert_frame_equal(X_train, mock_comparative_models_return[3])
        pd.testing.assert_frame_equal(X_test, mock_comparative_models_return[4])
        pd.testing.assert_series_equal(y_test, mock_comparative_models_return[5])


    @staticmethod
    def test_get_model_calls_comparative_models(monkeypatch, sample_df):
        """Test get_model calls comparative_models with correct arguments."""

        mock_func = mock.Mock(return_value=({}, {}, None, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int')))
        monkeypatch.setattr("src.telco_customer_churn_analysis.telco_customer_churn_analysis.comparative_models", mock_func)

        get_model(sample_df)

        mock_func.assert_called_once()
        args, kwargs = mock_func.call_args

        assert args[0].equals(sample_df)

        assert args[1] == 'Churn Value'

        assert isinstance(args[2], dict)
        assert 'KNeighbors' in args[2]
        assert 'RandomForest' in args[2]
        assert 'GaussianNB' in args[2]

        assert kwargs.get('metric', 'recall') == 'recall'


    @staticmethod
    @pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
    def test_get_model_handles_exceptions(monkeypatch, sample_df, exception_type):
        """Test get_model raises exceptions from comparative_models."""

        def raise_exception(*args, **kwargs):
            raise exception_type("Error!")

        monkeypatch.setattr("src.telco_customer_churn_analysis.telco_customer_churn_analysis.comparative_models", raise_exception)

        with pytest.raises(exception_type):
            get_model(sample_df)


## global_explainer tests

class TestGlobalExplainer:
    @staticmethod
    def test_global_explainer_calls_model_global_explainer(simple_pipeline):
        """Test global_explainer calls model_global_explainer with correct args."""

        pipe, X = simple_pipeline
        
        with mock.patch("src.telco_customer_churn_analysis.telco_customer_churn_analysis.model_global_explainer") as mock_explainer:
            global_explainer(pipe, X, X)
            mock_explainer.assert_called_once_with(pipe, X, X)


    @staticmethod
    def test_global_explainer_raises_for_non_pipeline(sample_df):
        """Test global_explainer raises error for non-pipeline input."""

        model = RandomForestClassifier()
        X = sample_df[['Monthly Charges', 'Total Charges', 'Tenure Months', 'City', 'Gender']]
        
        with pytest.raises(ValueError, match="must be a sklearn.pipeline.Pipeline object"):
            global_explainer(model, X, X)


    @staticmethod
    def test_global_explainer_raises_missing_steps(simple_pipeline, sample_df):
        """Test global_explainer raises error if pipeline missing required steps."""

        pipe = simple_pipeline[0]
        pipe_no_classifier = Pipeline(pipe.steps[:-1])
        
        with pytest.raises(ValueError, match="Pipeline must contain steps named 'preprocessor' and 'classifier'"):
            global_explainer(pipe_no_classifier, sample_df, sample_df)


    @staticmethod
    def test_global_explainer_raises_missing_get_feature_names_out(monkeypatch, simple_pipeline, sample_df):
        """Test global_explainer raises error if preprocessor lacks get_feature_names_out."""

        pipe, X = simple_pipeline
        pipe.named_steps['preprocessor'].get_feature_names_out = None
        
        with pytest.raises(AttributeError, match="Failed to retrieve feature names"):
            global_explainer(pipe, X, X)


    @staticmethod
    def test_global_explainer_runs_without_errors(simple_pipeline, sample_df):
        """Test global_explainer runs without errors on valid pipeline."""

        pipe, X = simple_pipeline
        global_explainer(pipe, X, X)


## local_explainer tests

class TestLocalExplainer:
    @staticmethod
    def test_local_explainer_runs(simple_pipeline, sample_df):
        """Test local_explainer runs and shows plot for valid pipeline and index."""

        pipe, X = simple_pipeline
        X_train = X.copy()
        X_test = X.copy()

        with mock.patch('matplotlib.pyplot.show') as mock_show:
            local_explainer(pipe, X_train, X_test, index=0)
            mock_show.assert_called_once()


    @staticmethod
    def test_local_explainer_index_out_of_bounds(simple_pipeline, sample_df):
        """Test local_explainer raises ValueError if index is out of bounds."""

        pipe, X = simple_pipeline
        X_train = X.copy()
        X_test = X.copy()

        with pytest.raises(ValueError, match="Index 100 is out of bounds"):
            local_explainer(pipe, X_train, X_test, index=100)


    @staticmethod
    def test_local_explainer_invalid_model_type(sample_df):
        """Test local_explainer raises ValueError for invalid model type."""

        X_train = sample_df.copy()
        X_test = sample_df.copy()

        class DummyModel:
            pass

        dummy_model = DummyModel()

        with pytest.raises(ValueError, match="must be a sklearn.pipeline.Pipeline"):
            local_explainer(dummy_model, X_train, X_test)


    @staticmethod
    def test_local_explainer_missing_steps(sample_df):
        """Test local_explainer raises error if pipeline missing required steps."""

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        preprocessor = StandardScaler()
        clf = RandomForestClassifier()

        pipe = Pipeline([
            ('some_step', preprocessor),
            ('another_step', clf)
        ])

        X_train = sample_df.copy()
        X_test = sample_df.copy()

        with pytest.raises(ValueError, match="Pipeline must contain steps named 'preprocessor' and 'classifier'"):
            local_explainer(pipe, X_train, X_test)


    @staticmethod
    def test_local_explainer_preprocessor_no_feature_names(sample_df):
        """Test local_explainer raises error if preprocessor lacks get_feature_names_out."""

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        class BadPreprocessor:
            def transform(self, X):
                return X.values

        bad_preprocessor = BadPreprocessor()
        clf = RandomForestClassifier()
        pipe = Pipeline([
            ('preprocessor', bad_preprocessor),
            ('classifier', clf)
        ])

        X_train = sample_df.copy()
        X_test = sample_df.copy()

        with pytest.raises(AttributeError, match="Failed to retrieve feature names"):
            local_explainer(pipe, X_train, X_test)


## profit_curve_threshold

class TestProfitCurveThreshold:
    @staticmethod
    def test_profit_curve_threshold_success():
        """Test profit_curve_threshold returns expected threshold value."""

        model_df = pd.DataFrame({
            'predictions_proba': [np.array([0.2, 0.4, 0.6, 0.8, 0.9])]
        })
        y_test = pd.Series([0, 1, 1, 0, 1])

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_aggfunc', return_value=2500), \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.profit_curve', return_value=(0.6, 1500.0, {0.6: 1500.0})):

            threshold = profit_curve_threshold(
                aggfunc='median',
                col='CLTV',
                model_df=model_df,
                cost=100,
                retention_rate=0.7,
                y_test=y_test
            )

            assert isinstance(threshold, float)
            assert threshold == 0.6


    @staticmethod
    def test_profit_curve_threshold_df_aggfunc_error():
        """Test profit_curve_threshold raises ValueError if df_aggfunc fails."""

        model_df = pd.DataFrame({
            'predictions_proba': [np.array([0.1, 0.2, 0.3])]
        })
        y_test = pd.Series([0, 1, 0])

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_aggfunc', side_effect=ValueError("Invalid aggfunc")):
            with pytest.raises(ValueError, match="Invalid aggfunc"):
                profit_curve_threshold(
                    aggfunc='bad_func',
                    col='CLTV',
                    model_df=model_df,
                    cost=50,
                    retention_rate=0.5,
                    y_test=y_test
                )


    @staticmethod
    def test_profit_curve_threshold_profit_curve_fails():
        """Test profit_curve_threshold raises ValueError if profit_curve fails."""

        model_df = pd.DataFrame({
            'predictions_proba': [np.array([0.1, 0.2, 0.3])]
        })
        y_test = pd.Series([0, 1, 0])

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_aggfunc', return_value=2000), \
            mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.profit_curve', side_effect=ValueError("Profit calculation failed")):

            with pytest.raises(ValueError, match="Profit calculation failed"):
                profit_curve_threshold(
                    aggfunc='median',
                    col='CLTV',
                    model_df=model_df,
                    cost=100,
                    retention_rate=0.5,
                    y_test=y_test
                )


    @staticmethod
    def test_profit_curve_threshold_missing_column():
        """Test profit_curve_threshold raises error if column is missing in DataFrame."""

        model_df = pd.DataFrame({
            'predictions_proba': [np.array([0.2, 0.5, 0.8])]
        })
        y_test = pd.Series([0, 1, 0])

        def dummy_df_aggfunc(df, aggfunc, col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            return 1000

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.df_aggfunc', side_effect=dummy_df_aggfunc):
            with pytest.raises(ValueError, match="Column 'Missing Column' not found in DataFrame"):
                profit_curve_threshold(
                    aggfunc='median',
                    col='Missing Column',
                    model_df=model_df,
                    cost=50,
                    retention_rate=0.5,
                    y_test=y_test
                )


## deploy_model tests

class TestDeployModel:
    @staticmethod
    def test_deploy_model_success(sample_df, simple_pipeline):
        """Test deploy_model completes successfully and calls deployment_model."""

        model, _ = simple_pipeline
        target = 'Churn Value'
        cols_to_drop = ['City', 'CLTV', 'Churn Reason']

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.deployment_model') as mock_deploy:
            mock_deploy.return_value = model
            result = deploy_model(sample_df, model, target, cols_to_drop)
            assert mock_deploy.called
            assert result is None


    @staticmethod
    def test_deploy_model_keyerror(sample_df, simple_pipeline):
        """Test deploy_model raises KeyError when specified column is missing."""

        model, _ = simple_pipeline
        target = 'Churn Value'
        cols_to_drop = ['NonExistentColumn']

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.deployment_model') as mock_deploy:
            mock_deploy.side_effect = KeyError("Column not found")

            with pytest.raises(KeyError, match="Column not found"):
                deploy_model(sample_df, model, target, cols_to_drop)


    @staticmethod
    def test_deploy_model_missing_classifier(sample_df):
        """Test deploy_model raises error if pipeline missing classifier step."""

        pipeline = Pipeline([
            ('preprocessor', OneHotEncoder())
        ])

        target = 'Churn Value'
        cols_to_drop = ['City','CLTV']

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.deployment_model') as mock_deploy:
            mock_deploy.side_effect = ValueError("Input model pipeline must contain a 'classifier' step.")
            
            with pytest.raises(ValueError, match="must contain a 'classifier'"):
                deploy_model(sample_df, pipeline, target, cols_to_drop)


    @staticmethod
    def test_deploy_model_preprocess_fallback(sample_df, simple_pipeline):
        """Test deploy_model falls back gracefully when preprocessing fails."""

        model, _ = simple_pipeline
        target = 'Churn Value'
        cols_to_drop = ['City','CLTV']

        with mock.patch('src.telco_customer_churn_analysis.telco_customer_churn_analysis.preprocess_data', side_effect=Exception("Preprocessing failed")), \
            mock.patch('src.telco_customer_churn_analysis.model_utils.joblib.dump'), \
            mock.patch('src.telco_customer_churn_analysis.model_utils.clone', side_effect=lambda x: x):

            result = deploy_model(sample_df, model, target, cols_to_drop)
            assert result is None


## generate_test_data tests

class TestGenerateTestData:
    @staticmethod
    def test_generate_test_data_returns_dataframe():
        """Test generate_test_data returns a DataFrame."""

        df = generate_test_data()
        assert isinstance(df, pd.DataFrame), "Output is not a DataFrame"


    @staticmethod
    def test_generate_test_data_has_expected_columns():
        """Test generate_test_data contains all expected columns."""

        df = generate_test_data()

        expected_columns = [
            'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long',
            'Latitude', 'Longitude', 'Gender', 'Senior Citizen', 'Partner',
            'Dependents', 'Tenure Months', 'Phone Service', 'Multiple Lines',
            'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
            'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract',
            'Paperless Billing', 'Payment Method', 'Monthly Charges', 'Total Charges',
            'Tenure Group'
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"


    @staticmethod
    def test_generate_test_data_has_non_empty_rows():
        """Test generate_test_data produces non-empty DataFrame with 10,000 rows."""

        df = generate_test_data()
        assert not df.empty, "Generated data is empty"
        assert len(df) == 10000, "Expected 10,000 rows of data"


    @staticmethod
    def test_generate_test_data_tenure_group_values():
        """Test generate_test_data Tenure Group column contains expected values."""

        df = generate_test_data()
        assert 'Tenure Group' in df.columns, "Tenure Group column is missing"

        expected_groups = [
            'New Customer', 'New/Established Customer',
            'Established/Veteran Customer', 'Veteran Customer'
        ]

        actual_groups = df['Tenure Group'].unique()
        for group in expected_groups:
            assert group in actual_groups, f"Missing Tenure Group: {group}"


## predict_df tests

class TestPredictDf:
    @staticmethod
    def test_predict_df_valid_output(monkeypatch, mock_prediction_df, mock_model):
        """Test predict_df returns valid DataFrame with expected columns and values."""

        monkeypatch.setattr("src.telco_customer_churn_analysis.model_utils.joblib.load", lambda path: mock_model)

        result_df = predict_df(mock_prediction_df, threshold=0.5)

        assert isinstance(result_df, pd.DataFrame)

        assert 'Churn Probability' in result_df.columns
        assert 'Intervention Flag' in result_df.columns

        assert len(result_df) == 2
        assert result_df.iloc[0]['Contract'] == 'Month-to-month'
        assert result_df.iloc[0]['Churn Probability'] == 0.8
        assert result_df.iloc[0]['Intervention Flag'] == 1



    @staticmethod
    def test_predict_df_threshold_behavior(monkeypatch, mock_prediction_df, mock_model):
        """Test predict_df sets intervention flags correctly for different thresholds."""

        monkeypatch.setattr("src.telco_customer_churn_analysis.model_utils.joblib.load", lambda path: mock_model)

        result_df = predict_df(mock_prediction_df, threshold=0.9)
        assert (result_df['Intervention Flag'] == 1).sum() == 0, "No customers should meet 0.9 threshold"

        result_df = predict_df(mock_prediction_df, threshold=0.1)
        assert (result_df['Intervention Flag'] == 1).sum() >= 1
        assert result_df[result_df['Intervention Flag'] == 1].shape[0] == 2


    @staticmethod
    def test_predict_df_invalid_threshold(monkeypatch, mock_prediction_df, mock_model):
        """Test predict_df raises ValueError for invalid threshold values."""

        monkeypatch.setattr("src.telco_customer_churn_analysis.model_utils.joblib.load", lambda path: mock_model)

        with pytest.raises(ValueError):
            predict_df(mock_prediction_df, threshold=1.5)


    @staticmethod
    def test_predict_df_model_load_failure(monkeypatch, mock_prediction_df):
        """Test predict_df returns empty DataFrame if model fails to load."""

        monkeypatch.setattr("src.telco_customer_churn_analysis.model_utils.joblib.load", lambda path: (_ for _ in ()).throw(FileNotFoundError()))

        result_df = predict_df(mock_prediction_df)
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty


    @staticmethod
    def test_predict_df_model_predict_failure(monkeypatch, mock_prediction_df):
        """Test predict_df returns empty DataFrame if model prediction fails."""

        broken_model = mock.MagicMock()
        broken_model.predict_proba.side_effect = Exception("Prediction failed")

        monkeypatch.setattr("src.telco_customer_churn_analysis.model_utils.joblib.load", lambda path: broken_model)

        result_df = predict_df(mock_prediction_df)
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty


## abc_test_assignment

class TestAbcTestAssignment:
    @staticmethod
    def test_abc_test_assignment_adds_columns(sample_df):
        """Test abc_test_assignment correctly adds Group and Intervention Details columns."""

        high_risk_df = sample_df[sample_df['Contract'] == 'Month-to-month'].copy()
        high_risk_df['Intervention Flag'] = np.random.choice([0, 1])

        result = abc_test_assignment(high_risk_df)

        assert 'Group' in result.columns
        assert 'Intervention Details' in result.columns

        valid_groups = {'A (Control)', 'B (Price Offer)', 'C (Service Offer)'}
        assert set(result['Group']).issubset(valid_groups)

        mapping = {
            'A (Control)': 'None (Control)',
            'B (Price Offer)': '10% Off 1-Year Contract',
            'C (Service Offer)': '6 Months Free Tech Support',
        }
        for group, intervention in mapping.items():
            assert all(result.loc[result['Group'] == group, 'Intervention Details'] == intervention)


    @staticmethod
    def test_abc_test_assignment_empty_df_returns_empty():
        """Test abc_test_assignment returns empty DataFrame if input is empty."""

        empty_df = pd.DataFrame()
        result = abc_test_assignment(empty_df)
        assert result.empty


    @staticmethod
    def test_abc_test_assignment_preserves_input_data_columns(sample_df):
        """Test abc_test_assignment preserves input data columns."""

        high_risk_df = sample_df[sample_df['Contract'] == 'Month-to-month'].copy()
        high_risk_df['Intervention Flag'] = 1

        result = abc_test_assignment(high_risk_df)

        if not result.empty:
            for col in high_risk_df.columns:
                assert col in result.columns
        else:
            assert isinstance(result, pd.DataFrame)


    @staticmethod
    def test_abc_test_assignment_random_seed_consistency(sample_df):
        """Test abc_test_assignment produces consistent results with random assignment."""

        high_risk_df = sample_df[sample_df['Contract'] == 'Month-to-month'].copy()
        high_risk_df['Intervention Flag'] = np.random.choice([0, 1])

        result1 = abc_test_assignment(high_risk_df)
        result2 = abc_test_assignment(high_risk_df)

        pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))


    @staticmethod
    def test_abc_test_assignment_raises_keyerror_if_columns_missing():
        """Test abc_test_assignment raises KeyError if required columns are missing."""

        df = pd.DataFrame({'SomeColumn': [1, 2]})
        df['Intervention Flag'] = np.random.choice([0, 1])
        df['Contract'] = 'Month-to-month'


        result = abc_test_assignment(df)
        assert 'Group' in result.columns
        assert 'Intervention Details' in result.columns


    @staticmethod
    def test_abc_test_assignment_empty_input_prints_message(capfd):
        """Test abc_test_assignment prints message and returns empty DataFrame if input empty."""

        empty_df = pd.DataFrame()
        result = abc_test_assignment(empty_df)
        out, _ = capfd.readouterr()
        assert "Dataframe empty" in out
        assert result.empty