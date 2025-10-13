import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unittest import mock
import os
import random
from pathlib import Path

from src.telco_customer_churn_analysis.utils import (safe_display, read_excel, df_head,
                                                     col_replace, null_rows, df_loc, df_aggfunc,
                                                     drop_labels, count_plot, histogram, heatmap,
                                                     bin_and_plot, chi_squared_test, generate_data)
                    

@pytest.fixture
def sample_df():
    """Provides a sample DataFrame for testing."""
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': ['x', 'y', 'z', 'x', 'y'],
        'C': [1.1, 2.2, np.nan, 3.3, np.nan],
        'D': pd.to_datetime(['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01'])
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_df():
    """Provides an empty DataFrame for testing."""
    return pd.DataFrame({'A': [], 'B': [], 'C': [], 'D': []})


@pytest.fixture
def mock_excel_file(tmp_path):
    """Creates a temporary dummy Excel file for testing"""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': ['2021-01-01', '2021-02-01']})
    file_path = tmp_path / "test.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)


@pytest.fixture(autouse=True)
def mock_plotting_calls(monkeypatch):
    """Mocks `plt.show()` and `plt.subplots()` to prevent plot display and enable checks."""
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'tight_layout', lambda: None)

    mock_fig = mock.Mock()
    mock_ax = mock.Mock()
    mock_ax.set_xticklabels.return_value = None
    mock_ax.get_xticklabels.return_value = []
    mock_ax.set_yticklabels.return_value = None
    mock_ax.get_yticklabels.return_value = []
    mock_ax.get_figure.return_value = mock_fig

    mock_ax.containers = [mock.Mock()]
    mock_ax.bar_label.return_value = None

    def mock_subplots(*args, **kwargs):
        return (mock_fig, mock_ax)
    
    monkeypatch.setattr(plt, 'subplots', mock_subplots)

    mock_countplot = mock.Mock()
    monkeypatch.setattr(sns, 'countplot', mock_countplot)

    mock_histplot = mock.Mock()
    monkeypatch.setattr(sns, 'histplot', mock_histplot)

    yield mock_ax, mock_countplot, mock_histplot


@pytest.fixture(autouse=True)
def mock_heatmap_setup(mock_plotting_calls, monkeypatch):
    mock_fig = mock.Mock()
    mock_ax = mock.Mock()

    monkeypatch.setattr(plt, 'figure', mock.Mock())
    monkeypatch.setattr(plt, 'title', mock.Mock())
    monkeypatch.setattr(plt, 'yticks', mock.Mock())
    monkeypatch.setattr(plt, 'xticks', mock.Mock())
    monkeypatch.setattr(plt, 'tight_layout', lambda: None)
    monkeypatch.setattr(plt, 'show', lambda: None)

    mock_heatmap = mock.Mock()
    monkeypatch.setattr(sns, 'heatmap', mock_heatmap)

    mock_corr_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'C': [0.5, 1.0]
    }, index=['A', 'C'])

    def mock_df_corr(numeric_only):
        return mock_corr_matrix
    
    yield mock_heatmap, mock_df_corr


@pytest.fixture
def mock_count_plot_dependency(monkeypatch):
    mock_count_plot = mock.Mock()
    import src.telco_customer_churn_analysis.utils as path_count_plot

    monkeypatch.setattr(path_count_plot, 'count_plot', mock_count_plot)

    yield mock_count_plot


@pytest.fixture
def sample_df_chi2():
    data = {
        'A': [1, 2, 3, 4, 5] * 10,
        'B': ['x', 'y'] * 25,
        'C': ['P', 'Q', 'P', 'Q', 'R'] * 10,
        'D': np.random.choice(['Yes', 'No'], size=50, p=[0.7, 0.3])
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_chi2_independence(monkeypatch):
    ## Scenario 1 (Significant Result - Moderate Association)
    expected_sig = pd.DataFrame({'x': [5.0, 5.0], 'y': [5.0, 5.0]}, index=['P', 'Q'])
    observed_sig = pd.DataFrame({'x': [0, 2], 'y': [2, 8]}, index=['P', 'Q'])
    stats_sig = pd.DataFrame({
        'test': ['pearson'],
        'chisq': [12.8],
        'dof': [1],
        'pval': [0.0003],
        'cramer': [0.30],
        'power': [0.95]
    })


    ## Scenario 2 (Non-Significant Result)
    expected_non_sig = pd.DataFrame({'x': [10, 10], 'y': [10, 10]}, index=['P', 'Q'])
    observed_non_sig = expected_non_sig.copy()
    stats_non_sig = stats_sig.copy()
    stats_non_sig.loc[0, 'pval'] = 0.50
    stats_non_sig.loc[0, 'cramer'] = 0.05
    stats_non_sig.loc[0, 'power'] = 0.15


    ## Scenario 3 (Assumption violated (expected < 5 cells > 20%))
    expected_viol = pd.DataFrame({'x': [1, 1], 'y': [8, 1]}, index=['P', 'Q'])
    observed_viol = expected_viol.copy()

    def mock_chi2(data, x, y):
        if x == 'B' and y == 'C':
            return expected_sig, observed_sig, stats_sig
        
        elif x == 'B' and y == 'A':
            return expected_non_sig, observed_non_sig, stats_non_sig
        
        elif x == 'B' and y == 'D':
            return expected_viol, observed_viol, stats_sig
        
        else:
            raise RuntimeError("Mock called with unexpected columns.")
        
    with mock.patch('pingouin.chi2_independence', side_effect=mock_chi2) as m:
        yield m


@pytest.fixture
def capfd_out(capfd):
    yield capfd

## safe_display tests

class TestSafeDisplay:
    @staticmethod
    def test_safe_display_uses_display_when_available(sample_df):
        """Test that safe_display uses IPython.display.display when available."""
        mock_display_func = mock.MagicMock()
        mock_display_module = mock.MagicMock(display=mock_display_func)

        with mock.patch.dict('sys.modules', {'IPython.display': mock_display_module}):
            with mock.patch('src.telco_customer_churn_analysis.utils.display', mock_display_func):
                result = safe_display(sample_df)
        
        mock_display_func.assert_called_once_with(sample_df)
        assert result.equals(sample_df)


    @staticmethod
    @mock.patch('builtins.print')
    def test_safe_display_falls_back_to_print_on_nameerror(mock_print, sample_df):
        """Test safe display falls back to `print()` when `display()` is not defined (NameError)."""
        with mock.patch('src.telco_customer_churn_analysis.utils.display', side_effect=NameError("Name 'display' not defined")):
            safe_display(sample_df)

        mock_print.assert_called_once_with(sample_df)


    @staticmethod
    @mock.patch('builtins.print')
    def test_safe_display_falls_back_to_print_on_general_exception(mock_print, sample_df):
        """Test `safe_display()` falls back to `print()` and prints an error message on other Exceptions."""
        error_msg = "Test Display Error: Cannot render widget."

        with mock.patch('src.telco_customer_churn_analysis.utils.display', side_effect=ValueError(error_msg)):
            safe_display(sample_df)

        assert mock_print.call_count == 2

        call_1_arg = mock_print.call_args_list[0][0][0]
        call_2_arg = mock_print.call_args_list[1][0][0]

        error_prefix = "Error: Failed to display data. Falling back to print()."

        is_df_call_1 = isinstance(call_1_arg, pd.DataFrame) and call_1_arg.equals(sample_df)
        is_df_call_2 = isinstance(call_2_arg, pd.DataFrame) and call_2_arg.equals(sample_df)

        assert is_df_call_1 or is_df_call_2, "Expected one print call to contain the fallabck DataFrame."

        is_err_call_1 = isinstance(call_1_arg, str) and error_prefix in call_1_arg
        is_err_call_2 = isinstance(call_2_arg, str) and error_prefix in call_2_arg

        assert (is_df_call_1 and is_err_call_2) or (is_err_call_1 and is_df_call_2), f"Expected one call to be the DataFrame and the other to contain the error prefix: '{error_prefix}'."


## read_excel test

class TestReadExcel:
    @staticmethod
    def test_read_excel_success(mock_excel_file):
        """Test successful reading a valid Excel file without date parsing."""
        df = read_excel(mock_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert df['A'].dtype == 'int64'
        assert df['C'].dtype == 'object'

    
    @staticmethod
    def test_read_excel_with_date_parsing(mock_excel_file):
        """Test successful reading an Excel file with date parsing."""
        df = read_excel(mock_excel_file, date_cols=['C'])
        assert df['C'].dtype == 'datetime64[ns]'

    
    @staticmethod
    def test_read_excel_raises_file_not_found_error():
        """Test that `read_excel()` raises FileNotFound for a non-existing file."""
        non_existent_path = "non_existent_path.xlsx"
        with pytest.raises(FileNotFoundError, match=f"Error: File not found at path: {non_existent_path}"):
            read_excel(non_existent_path)


    @staticmethod
    @mock.patch('pandas.read_excel')
    def test_read_excel_raises_import_error(mock_read_excel):
        """Test that `read_excel()` raises ImportError when the Excel engine is missing."""
        mock_read_excel.side_effect = ImportError("No engine 'openpyxl' or 'xlrd' found")
        with pytest.raises(ImportError, match="Error. Missing Excel engine."):
            read_excel("mock_path.xlsx")


    @staticmethod
    @mock.patch('pandas.read_excel')
    @mock.patch('builtins.print')
    def test_read_excel_warns_on_empty_df(mock_print, mock_read_excel):
        """Test that `read_excel()` warns when an empty DataFrame is returned."""
        mock_path = "mock_path.xlsx"
        mock_read_excel.return_value = pd.DataFrame()
        df = read_excel(mock_path)

        mock_print.assert_called_once_with(f"Warning: File loaded successfully, but '{mock_path}' is empty")
        assert df.empty


    @staticmethod
    @mock.patch('pandas.read_excel')
    def test_read_excel_raises_runtime_error_on_general_exception(mock_read_excel):
        """Test that `read_excel()` raises a generic RuntimeError for unhandled errors."""
        mock_read_excel.side_effect = Exception("General parsi8ng failure")
        with pytest.raises(RuntimeError, match="A critical error occurred while reading or parsing the Excel file"):
            read_excel("mock_path.xlsx")


## df_head tests

class TestDfHead:
    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.utils.safe_display', side_effect=lambda x: x)
    @mock.patch('builtins.print')
    def test_df_head_returns_correct_tuple(mock_print, mock_safe_display, sample_df):
        """Test that `df_head()` returns a tuple of (head, describe) DataFrames with correct content."""
        head, describe = df_head(sample_df)

        assert isinstance(head, pd.DataFrame)
        assert isinstance(describe, pd.DataFrame)
        assert head.shape == (5, 4)
        assert describe.shape == (8, 2)
        assert mock_safe_display.call_count == 2


    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.utils.safe_display', side_effect=lambda x: x)
    @mock.patch('builtins.print')
    def test_df_head_with_custom_n_rows(mock_print, mock_safe_display, sample_df):
        """Test that `df_head()` returns correct number of rows when n_rows is specified."""
        head, _ = df_head(sample_df, n_rows=3)
        
        assert head.shape == (3, 4)


    @staticmethod
    def test_df_head_raises_type_error():
        """Test that `df_head()` raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame, but received list"):
            df_head([1, 2, 3])


    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.utils.safe_display', side_effect=lambda x: x)
    @mock.patch('builtins.print')
    def test_df_head_invalid_n_rows_warning_and_default(mock_print, mock_safe_display, sample_df):
        """Test that `df_head()` warns when n_rows is invalid and uses the default 5."""
        head, _ = df_head(sample_df, n_rows=-1)

        mock_print.assert_any_call("Warning: n_rows must be a non-negative integer. Defaulting to 5.")

        assert head.shape == (5, 4)


    @staticmethod
    @mock.patch('src.telco_customer_churn_analysis.utils.safe_display', side_effect=lambda x: x)
    def test_df_head_raises_runtime_error_on_df_head_failure(mock_safe_display, sample_df):
        """Test that `df_head()` raises RuntimeError if df.head() fails."""
        with mock.patch.object(sample_df, 'head', side_effect=Exception("Head failed")):
            with pytest.raises(RuntimeError, match="Error occurred while processing df.head"):
                df_head(sample_df)


## col_replace tests

class TestColReplace:
    @staticmethod
    def test_col_replace_success_single_value(sample_df):
        """Test successful replacement of single value."""
        new_df = col_replace(sample_df, 'B', 'x', 'new_x')

        assert 'x' not in new_df['B'].values
        assert 'new_x' in new_df['B'].values
        assert new_df.shape == sample_df.shape
        assert new_df.loc[0, 'B'] == 'new_x'

    
    @staticmethod
    def test_col_replace_success_list_of_values(sample_df):
        """Test successful replacement of a list of values"""
        new_df = col_replace(sample_df, 'B', ['x', 'y'], 'new_xy')

        assert 'x' not in new_df['B'].values
        assert 'y' not in new_df['B'].values
        assert 'new_xy' in new_df['B'].values
        assert new_df.loc[0, 'B'] == 'new_xy'
        assert new_df.loc[1, 'B'] == 'new_xy'


    @staticmethod
    def test_col_replace_raises_type_error():
        """Test that `col_replace()` raises TypeError when input 'df' is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame, but received list"):
            col_replace([1 ,2], 'A', 1, 0)


    @staticmethod
    def test_col_replace_raises_key_error(sample_df):
        """Test that `col_replace()` raises KeyError when the column is missing."""
        with pytest.raises(KeyError, match="Column 'Z' not found in DataFrame"):
            col_replace(sample_df, 'Z', 1, 0)


    @staticmethod
    @mock.patch('builtins.print')
    def test_col_replace_wars_on_no_occurrence(mock_print, sample_df):
        """Test that `col_replace()` warns when the 'old_var' is not found."""
        col_replace(sample_df, 'B', 'non_existent', 'new_value')
        mock_print.assert_called_once_with("Warning: No occurrence of non_existent found in column 'B'.")


    @staticmethod
    @mock.patch('builtins.print')
    def test_col_replace_on_no_occurrence_list(mock_print, sample_df):
        """Test that `col_replace()` warns when none of the 'old_var' list elements are found."""
        col_replace(sample_df, 'B', ['non_existent_1', 'non_existent_2'], 'new_value')
        mock_print.assert_called_once()


    @staticmethod
    def test_col_replace_raises_runtime_error_on_replacement_failure(sample_df):
        """Test that `col_replace()` raises a RuntimeError for unexpected replacement issues."""
        with mock.patch('pandas.Series.replace', side_effect=Exception("Replacement failed")) as mock_replace:
            with pytest.raises(RuntimeError, match="An unexpected error occurred during replacement in column 'B'."):
                col_replace(sample_df, 'B', 'x', 'new_x')

        mock_replace.assert_called_once()


## null_rows tests

class TestNullRows:
    @staticmethod
    def test_null_rows_returns_dataframe_for_all_cols(sample_df):
        """Test `null_rows()` returns a DataFrame of booleans when no columns are specified."""
        result = null_rows(sample_df)

        expected_data = {
            'A': [False, False, True, False, False],
            'B': [False, False, False, False, False],
            'C': [False, False, True, False, True],
            'D': [False, False, False, False, False]
        }

        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result, expected_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_df.shape


    @staticmethod
    def test_null_rows_returns_series_for_single_col(sample_df):
        """Test `null_rows()` returns a Series of booleans when one column is specified."""
        result = null_rows(sample_df, 'A')

        expected_data = [False, False, True, False, False]
        expected_series = pd.Series(expected_data, name='A')

        pd.testing.assert_series_equal(result, expected_series)
        assert isinstance(result, pd.Series)
        assert result.shape[0] == sample_df.shape[0]


    @staticmethod
    def test_null_rows_returns_dataframe_for_multiple_cols(sample_df):
        """Test `null_rows()` returns a DataFrame when multiple columns are specified."""
        result = null_rows(sample_df, 'A', 'C')

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (sample_df.shape[0], 2)
        assert 'B' not in result.columns


    @staticmethod
    def test_null_rows_raises_type_error(sample_df):
        """Test that `null_rows()` raises a TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="TypeError: Input 'df' must be a pandas DataFrame, but received list"):
            null_rows([1, 2, 3], 'A')


    @staticmethod
    def test_null_rows_raises_key_error(sample_df):
        """Test that `null_rows()` raises KeyError when a specified column is missing."""
        missing_col = 'Z'

        expected_prefix = "KeyError: Column(s) not found."
        expected_missing_key_check = missing_col
        expected_available_cols_check = str(list(sample_df.columns))

        with pytest.raises(KeyError) as excinfo:
            null_rows(sample_df, 'A', missing_col)

        error_message_raw = str(excinfo.value)
        error_message = error_message_raw.strip('"\' ')

        assert error_message.startswith(expected_prefix), (f"Assertion Failed: Expected error message to start with '{expected_prefix}', but it starts with: {error_message[:len(expected_prefix) + 5]}. Full Message: {error_message_raw}")

        assert expected_missing_key_check in error_message, (f"Missing key '{expected_missing_key_check}' not found in error message: {error_message_raw}")

        assert expected_available_cols_check in error_message, (f"Available columns list '{expected_available_cols_check}' not found in error message: {error_message_raw}")


## df_loc tests

class TestDfLoc:
    @staticmethod
    def test_df_loc_success_multiple_columns(sample_df):
        """Test selection of multiple columns resulting in a DataFrame."""
        condition = sample_df['B'] == 'y'

        result = df_loc(sample_df, condition, ['A', 'C'])

        expected_data = {
            'A': {1: 2.0, 4: 5.0},
            'C': {1: 2.2, 4: np.nan}
        }

        expected_df = pd.DataFrame(expected_data).set_index(pd.Index([1, 4]))

        pd.testing.assert_frame_equal(result, expected_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)


    @staticmethod
    def test_df_loc_success_single_column(sample_df):
        """Test selection of a single column resulting in a Series."""
        condition = sample_df['A'].notna()

        result = df_loc(sample_df, condition, 'D')

        expected_data = sample_df['D'].iloc[[0, 1, 3, 4]]
        expected_series = pd.Series(expected_data, index=[0, 1, 3, 4], name='D')

        pd.testing.assert_series_equal(result, expected_series)
        assert isinstance(result, pd.Series)

    
    @staticmethod
    def test_df_loc_empty_result(sample_df):
        """Test a condition that returns zero rows."""
        condition = sample_df['A'] > 100
        result = df_loc(sample_df, condition, ['A', 'B'])

        expected_df = sample_df.loc[condition, ['A', 'B']]

        pd.testing.assert_frame_equal(result, expected_df)
        assert result.empty


    @staticmethod
    def test_df_loc_raises_type_error_df(sample_df):
        """Test if `df_loc()` raises TypeError when 'df' is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame"):
            df_loc("not a df", sample_df['A'] > 0, 'A')


    @staticmethod
    def test_df_loc_raises_key_error(sample_df):
        """Test if `df_loc()` raises KeyError when a specified column is missing."""
        missing_col = 'Z'
        with pytest.raises(KeyError, match=fr"KeyError: The specified column\(s\) were not found in the DataFrame"):
            df_loc(sample_df, sample_df['A'] > 0, ['missing_col', 'A'])


    @staticmethod
    def test_df_loc_raises_value_error(sample_df):
        """Test if `df_loc()` raises ValueError when non-boolean values are used as condition."""
        non_boolean_condition = ['True', 'False', 'True', 'False', 'True']
        with pytest.raises(ValueError, match="ValueError: Invalid index/condition provided."):
            df_loc(sample_df, non_boolean_condition, 'A')


    @staticmethod
    @mock.patch('builtins.print')
    def test_df_loc_warns_on_unconventional_condition(mock_print, sample_df):
        """Test that `df_loc()` print a warning for non-Series/list/ndarray condition types."""
        unconventional_condition = (True, False, True, False, True)
        
        with pytest.raises(RuntimeError):
            df_loc(sample_df, unconventional_condition, 'A')

        mock_print.assert_called_once_with(mock.ANY)
        call_arg = mock_print.call_args_list[0][0][0]
        assert "Warning: 'condition' type (tuple) may lead to runtime errors." in call_arg


## df_aggfunc tests

class TestDfAggFunc:
    @staticmethod
    def test_df_aggfunc_single_string_col(sample_df):
        """Test aggregation with a single string function on a single column"""
        result = df_aggfunc(sample_df, aggfunc='mean', col='A')

        expected_value = sample_df['A'].mean()

        assert isinstance(result, (float, np.float64))
        assert result == expected_value


    @staticmethod
    def test_df_aggfunc_list_string_cols(sample_df):
        """Test aggregation with a list of functions on multiple columns."""
        result = df_aggfunc(sample_df, aggfunc=['min', 'max'], col=['A', 'C'])

        expected_data = {
            'A': [sample_df['A'].min(), sample_df['A'].max()],
            'C': [sample_df['C'].min(), sample_df['C'].max()]
        }

        expected_df = pd.DataFrame(expected_data, index=['min', 'max'])

        pd.testing.assert_frame_equal(result, expected_df)
        assert result.shape == (2, 2)


    @staticmethod
    def test_df_aggfunc_custom_callable(sample_df):
        """Test aggreagtion with a custom callable function."""
        def range_func(s):
            return s.max() - s.min() if not s.empty else np.nan
        
        result = df_aggfunc(sample_df, aggfunc=range_func, col=['A', 'C'])

        expected_range_A = sample_df['A'].max() - sample_df['A'].min()
        expected_range_C = sample_df['C'].max() - sample_df['C'].min()

        expected_series = pd.Series([expected_range_A, expected_range_C], index=['A', 'C'], name='range_func')

        expected_series.name = range_func.__name__

        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, expected_series, check_names=False)


    @staticmethod
    def test_df_aggfunc_dict_aggfunc(sample_df):
        """Test aggregation with a dictionary mapping columns to functions."""
        result = df_aggfunc(sample_df, aggfunc={'A': 'sum', 'B': 'count', 'C': ['mean', 'median']})

        expected_df = sample_df.agg({'A': 'sum', 'B': 'count', 'C': ['mean', 'median']})

        pd.testing.assert_frame_equal(result, expected_df)


    @staticmethod
    def test_df_aggfunc_value_counts_single_col(sample_df):
        """Test 'value_counts' on a single column resulting in a Series."""
        result = df_aggfunc(sample_df, aggfunc='value_counts', col='B')
        expected_series = sample_df['B'].value_counts()

        pd.testing.assert_series_equal(result, expected_series)
        assert isinstance(result, pd.Series)


    @staticmethod
    def test_df_aggfunc_value_counts_multiple_columns(sample_df):
        """Test 'value_counts" on multiple columns resulting in a dictionary of Series."""
        result = df_aggfunc(sample_df, aggfunc='value_counts', col=['B', 'A'])

        expected_dict = {
            'B': sample_df['B'].value_counts(),
            'A': sample_df['A'].value_counts()
        }

        assert isinstance(result, dict)
        assert list(result.keys()) == ['B', 'A']
        pd.testing.assert_series_equal(result['B'], expected_dict['B'])
        pd.testing.assert_series_equal(result['A'], expected_dict['A'])


    @staticmethod
    def test_df_aggfunc_raises_type_error_df(sample_df):
        """Test if ``df_aggfunc()` raises TypeError when 'df' is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame"):
            df_aggfunc('not_a_df', aggfunc='mean', col='A')


    @staticmethod
    def test_df_aggfunc_raises_type_error_col_non_string(sample_df):
        """Test if `df_aggfunc()` raises TypeError when list for 'col' contains non-string elements."""
        with pytest.raises(TypeError, match="List for 'col' must contain only strings"):
            df_aggfunc(sample_df, aggfunc='mean', col=['A', 123])


    @staticmethod
    def test_df_aggfunc_raises_key_error_missing_col(sample_df):
        """Test if `df_aggfunc()` raises KeyError when a specified column is missing."""
        missing_col = 'Z'

        pattern = fr"KeyError: The specified column\(s\) were not found in the DataFrame: \['{missing_col}'\].*"

        with pytest.raises(KeyError, match=pattern):
            df_aggfunc(sample_df, aggfunc='sum', col=[missing_col, 'A'])


    @staticmethod
    def test_df_aggfunc_raises_value_error_aggfunc_incompatible(sample_df):
        """Test if ``df_aggfunc()` raises ValueError when 'aggfunc' is incompatible with the data type."""
        with pytest.raises(ValueError, match="The aggregation function 'mean' is not available or compatible with the selected data type"):
            df_aggfunc(sample_df, aggfunc='mean', col='B')


    @staticmethod
    def test_df_aggfunc_raises_value_error_value_counts_no_col(sample_df):
        """Test if `df_aggfunc()` raises ValueError when 'value_counts' is used without 'col'."""
        with pytest.raises(ValueError, match="'value_counts' requires one or more columns to be specified in 'col'"):
            df_aggfunc(sample_df, aggfunc='value_counts', col=None)


    @staticmethod
    def test_df_aggfunc_raises_runtime_error_value_counts_on_empty_df(empty_df):
        """Test if `df_aggfunc()` raises RuntimeError for 'value_counts' on an empty DataFrame."""
        try:
            df_aggfunc(empty_df, aggfunc='value_counts', col='B')

        except RuntimeError as e:
            assert "RuntimeError: Failed to perform 'value_counts' on column(s)" in str(e)
        
        except Exception:
            pass


## drop_labels tests

class TestDropLabels:
    @staticmethod
    def test_drop_labels_no_labels_returns_copy(sample_df):
        """Test that passing 'labels=None' returns a deep copy of the original DataFrame."""
        original_shape = sample_df.shape
        result_df = drop_labels(sample_df, labels=None)

        assert result_df is not sample_df
        assert result_df.shape == original_shape
        pd.testing.assert_frame_equal(result_df, sample_df)


    @staticmethod
    def test_drop_labels_single_column_by_str(sample_df):
        """Test dropping a single column label using string input."""
        result_df = drop_labels(sample_df, labels='B', axis='columns')

        expected_columns = ['A', 'C', 'D']

        assert list(result_df.columns) == expected_columns
        assert result_df.shape == (5, 3)


    @staticmethod
    def test_drop_labels_multiple_columns_by_list(sample_df):
        """Test dropping multiple columns labels using a list of strings."""
        result_df = drop_labels(sample_df, labels=['A', 'C'], axis=1)

        expected_columns = ['B', 'D']
        
        assert list(result_df.columns) == expected_columns
        assert result_df.shape == (5, 2)



    @staticmethod
    def test_drop_labels_rows_by_index(sample_df):
        """Test dropping rows (index labels) using integer 'axis=0'"""
        result_df = drop_labels(sample_df, labels=[1, 4], axis=0)

        expected_index = [0, 2, 3]

        assert list(result_df.index) == expected_index
        assert result_df.shape == (3, 4)


    @staticmethod
    def test_drop_labels_missing_labels_ignored(sample_df):
        """Test that dropping non-existent labels is ignored due to errors='ignore'."""
        result_df = drop_labels(sample_df, labels=['B', 'Missing'], axis='columns')

        expected_columns = ['A', 'C', 'D']

        assert list(result_df.columns) == expected_columns
        assert result_df.shape == (5, 3)


    @staticmethod
    def test_drop_labels_raises_type_error_df(sample_df):
        """Test if `drop_labels()` raises TypeError when 'df' is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame"):
            drop_labels('not a df', labels='A', axis=1)


    @staticmethod
    def test_drop_labels_raises_type_error_labels_non_string_or_int(sample_df):
        """Test if `drop_labels()` raises TypeError when 'labels' is a list containing non-strings."""
        with pytest.raises(TypeError, match="TypeError: All elements in 'labels' list must be strings or integers."):
            drop_labels(sample_df, labels=[1.0, 'A'], axis='columns')


    @staticmethod
    def test_drop_labels_raises_type_error_axis_invalid_type(sample_df):
        """Test if `drop_labels()` raises TypeError when 'axis' is not an int or str."""
        with pytest.raises(TypeError, match="'axis' must be an integer \\(0 or 1\\) or a string \\('index' or 'columns'\\)"):
            drop_labels(sample_df, labels='A', axis=3.0)


    @staticmethod
    def test_drop_labels_raises_value_error_axis_invalid_value(sample_df):
        """Test if `drop_labels()` raises ValueError when 'axis' is a valid type but invalid value."""
        pattern = r"ValueError: 'axis' must be one of {0, 1, 'index', 'columns'}, but received '2'"

        with pytest.raises(ValueError, match=pattern):
            drop_labels(sample_df, labels='A', axis=2)


## count_plot tests

class TestCountPlot:
    @staticmethod
    def test_count_plot_x_axis(sample_df, mock_plotting_calls):
        """Test standard vertical count plot with minimal required arguments."""
        mock_ax, mock_countplot, _ = mock_plotting_calls
        
        count_plot(title='Title', label='Label', df=sample_df, col='B', axis='x')

        mock_countplot.assert_called_once()
        called_kwargs = mock_countplot.call_args[1]

        assert called_kwargs['x'] == 'B'

        mock_ax.set_title.assert_called_once_with('Title')
        mock_ax.set_xlabel.assert_called_once_with('Label')


    @staticmethod
    def test_count_plot_y_axis(sample_df, mock_plotting_calls):
        """Test horizontal count plot orientation."""
        mock_ax, mock_countplot, _ = mock_plotting_calls

        count_plot(title='Title', label='Label', df=sample_df, col='B', axis='y')

        mock_countplot.assert_called_once()
        called_kwargs = mock_countplot.call_args[1]

        assert called_kwargs['y'] == 'B'

        mock_ax.set_title.assert_called_once_with('Title')
        mock_ax.set_ylabel.assert_called_once_with('Label')

    
    @staticmethod
    def test_count_plot_with_hue_and_order(sample_df, mock_plotting_calls):
        """Test count plot with 'hue' and 'order' parameters"""
        mock_ax, mock_countplot, _ = mock_plotting_calls

        df_copy = sample_df.dropna(subset=['A']).copy()
        df_copy['A'] = df_copy['A'].astype('int').astype('str')
        expected_order = ['x', 'y', 'z']

        count_plot(title='Title', label='Label', df=df_copy, col='B', axis='x', hue='A', order=expected_order)

        mock_countplot.assert_called_once()
        called_kwargs = mock_countplot.call_args[1]

        assert called_kwargs['hue'] == 'A'
        assert called_kwargs['order'] == expected_order

        mock_ax.set_title.assert_called_once_with('Title')
        mock_ax.set_xlabel.assert_called_once_with('Label')


    @staticmethod
    def test_count_plot_tick_rotation(sample_df, mock_plotting_calls):
        """Test count plot with a non-zero tick rotation."""
        mock_ax, mock_countplot, _ = mock_plotting_calls

        count_plot(title='Title', label='Label', df=sample_df, col='B', tick_rotation=45)

        mock_ax.set_xticklabels.assert_called_once()

    @staticmethod
    def test_count_plot_type_error_df(sample_df):
        """Test `count_plot()` raises TypeError when 'df' is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame"):
            count_plot(title='Title', label='Label', df='not a df', col='B')


    @staticmethod
    def test_count_plot_key_error_col(sample_df):
        """Test `count_plot()` raises a KeyError when 'col' is missing."""
        with pytest.raises(KeyError, match="Column 'Missing' not found in DataFrame"):
            count_plot(title='Title', label='Label', df=sample_df, col='Missing')


    @staticmethod
    def test_count_plot_key_error_hue(sample_df):
        """Test `count_plot()` raises a KeyError when 'hue' is missing."""
        with pytest.raises(KeyError, match="Hue column 'Missing' not found in DataFrame"):
            count_plot(title='Title', label='Label', df=sample_df, col='B', hue='Missing')


    @staticmethod
    def test_count_plot_value_error_axis(sample_df):
        """Test `count_plot()` raises a ValueError when 'axis' is invalid."""
        with pytest.raises(ValueError, match="'axis' must be 'x' or 'y'"):
            count_plot(title='Title', label='Label', df=sample_df, col='B', axis='Z')


    @staticmethod
    def test_count_plot_value_error_tick_rotation(sample_df):
        """Test `count_plot()` raises a ValueError when 'tick_rotation' is invalid."""
        with pytest.raises(ValueError, match="'tick_rotation' must be a number"):
            count_plot(title='Title', label='Label', df=sample_df, col='B', tick_rotation='45')

    
    @staticmethod
    def test_count_plot_type_error_non_string_args(sample_df):
        """Test `count_plot()` raises a TypeError when 'title' is not a string."""
        with pytest.raises(TypeError, match="'title', 'label', and 'col' must all be strings"):
            count_plot(title=123, label='Label', df=sample_df, col='B')


## histogram tests

class TestHistogram:
    @staticmethod
    def test_histogram_x_axis(sample_df, mock_plotting_calls):
        """Test standard vertical histogram with default settings."""
        mock_ax, _ , mock_histplot = mock_plotting_calls

        histogram(title='Title', label='Label', df=sample_df, col='A', bins=10, axis='x')

        mock_ax.set_title.assert_called_once_with('Title')
        mock_ax.set_xlabel.assert_called_once_with('Label')

        mock_histplot.assert_called_once()
        called_kwargs = mock_histplot.call_args[1]

        assert called_kwargs['x'] == 'A'
        assert called_kwargs['bins'] == 10
        assert called_kwargs['kde'] is False
        assert called_kwargs['hue'] is None
        assert called_kwargs['data'] is sample_df


    @staticmethod
    def test_histogram_y_axis_with_kde(sample_df, mock_plotting_calls):
        """Test horizontal histogram with 'kde' and label swapping."""
        mock_ax, _ , mock_histplot = mock_plotting_calls

        histogram(title='Title', label='Label', df=sample_df, col='A', bins=10, axis='y', kde=True)

        mock_ax.set_title.assert_called_once_with('Title')
        mock_ax.set_ylabel.assert_called_once_with('Label')

        mock_histplot.assert_called_once()
        called_kwargs = mock_histplot.call_args[1]

        assert called_kwargs['y'] == 'A'
        assert called_kwargs['bins'] == 10
        assert called_kwargs['kde'] is True
        assert called_kwargs['hue'] is None
        assert called_kwargs['data'] is sample_df


    @staticmethod
    def test_histogram_with_hue_and_bin_list(sample_df, mock_plotting_calls):
        """Test histogram with a categorical 'hue' and a custom list of bins."""
        mock_ax, _ , mock_histplot = mock_plotting_calls

        custom_bins = [0, 1, 2, 3, 4, 5]

        histogram(title='Title', label='Label', df=sample_df, col='A', bins=custom_bins, hue='B')

        mock_ax.set_title.assert_called_once_with('Title')
        mock_ax.set_xlabel.assert_called_once_with('Label')

        mock_histplot.assert_called_once()
        called_kwargs = mock_histplot.call_args[1]

        assert called_kwargs['x'] == 'A'
        assert called_kwargs['bins'] == custom_bins
        assert called_kwargs['hue'] == 'B'
        assert called_kwargs['data'] is sample_df


    @staticmethod
    def test_histogram_type_error_df(sample_df):
        """Test `histogram()` raises a TypeError when 'df' is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame"):
            histogram(title='Title', label='Label', df='not a df', col='A', bins=10)

    
    @staticmethod
    def test_histogram_key_error_col(sample_df):
        """Test `histogram()` raises a KeyError when 'col' is missing."""
        with pytest.raises(KeyError, match="Column 'Missing' not found in DataFrame"):
            histogram(title='Title', label='Label', df=sample_df, col='Missing', bins=10)


    @staticmethod
    def test_histogram_key_error_hue(sample_df):
        """Test `histogram()` raises a KeyError when 'hue' is missing."""
        with pytest.raises(KeyError, match="Hue column 'Missing' not found in DataFrame"):
            histogram(title='Title', label='Label', df=sample_df, col='A', bins=10, hue='Missing')

    
    @staticmethod
    def test_histogram_value_error_axis(sample_df):
        """Test `histogram()` raises a ValueError when 'axis' is invalid."""
        with pytest.raises(ValueError, match="'axis' must be 'x' or 'y'"):
            histogram(title='Title', label='Label', df=sample_df, col='A', bins=10, axis='z')

    
    @staticmethod
    def test_histogram_runtime_error_non_numeric(sample_df, mock_plotting_calls):
        """Test `histogram()` raises a RuntimeError when the column is non-numerical."""
        mock_ax, _ , mock_histplot = mock_plotting_calls

        mock_histplot.side_effect = ValueError("Could not convert column to numeric type")

        with pytest.raises(RuntimeError, match="Check if column 'col' is numerical"):
            histogram(title='Title', label='Label', df=sample_df, col='B', bins=10)


## heatmap test

class TestHeatmap:
    @staticmethod
    def test_heatmap_basic_functionality(sample_df, mock_heatmap_setup, monkeypatch):
        """Test basic heatmap call with default parameters."""
        mock_heatmap, mock_df_corr = mock_heatmap_setup

        monkeypatch.setattr(sample_df, 'corr', mock_df_corr)

        heatmap(title='Title', df=sample_df)

        mock_heatmap.assert_called_once()
        called_args, called_kwargs = mock_heatmap.call_args

        assert called_args[0].shape == (2, 2)

        assert called_kwargs['annot'] is True
        assert called_kwargs['cmap'] == 'coolwarm'
        assert called_kwargs['fmt'] == '.2f'
        assert called_kwargs['annot_kws']['fontsize'] == 7

        plt.title.assert_called_once_with('Title')
        plt.yticks.assert_called_once_with(rotation=0)
        plt.xticks.assert_called_once_with(rotation=90)


    @staticmethod
    def test_heatmap_custom_params(sample_df, mock_heatmap_setup, monkeypatch):
        """Test heatmap with custom 'annot', 'cmap', 'fontsize', and 'num_decimals'."""
        mock_heatmap, mock_df_corr = mock_heatmap_setup

        monkeypatch.setattr(sample_df, 'corr', mock_df_corr)

        heatmap(title='Title', df=sample_df, annot=False, cmap='viridis', fontsize=10, num_decimals=4)

        mock_heatmap.assert_called_once()
        called_kwargs = mock_heatmap.call_args[1]

        assert called_kwargs['annot'] is False
        assert called_kwargs['cmap'] == 'viridis'
        assert called_kwargs['fmt'] == '.4f'
        assert called_kwargs['annot_kws']['fontsize'] == 10
        plt.title.assert_called_once_with('Title')


    @staticmethod
    def test_heatmap_value_error_no_numeric_columns():
        """Test `heatmap()` raises ValueError when the DataFrame contains no numeric columns."""
        non_numeric_df = pd.DataFrame({
            'A': ['a', 'b'],
            'B': ['c', 'd']
        })

        with pytest.raises(ValueError, match="DataFrame contains no numeric columns"):
            heatmap(title='Title', df=non_numeric_df)


    @staticmethod
    def test_heatmap_type_error_title(sample_df):
        """Test `heatmap()`` raises TypeError when title is non-string."""
        with pytest.raises(TypeError, match="'title' must be a string"):
            heatmap(title=123, df=sample_df)


    @staticmethod
    def test_heatmap_value_error_negative_decimals(sample_df):
        """Test `heatmap()`` raises ValueError when 'num_decimals' is negative."""
        with pytest.raises(ValueError, match="'num_decimals' must be a non-negative integer"):
            heatmap(title='Title', df=sample_df, num_decimals=-1)


    @staticmethod
    def test_heatmap_value_error_non_positive_fontsize(sample_df):
        """Test `heatmap()`` raises ValueError when 'fontsize' is non-positive."""
        with pytest.raises(ValueError, match="'fontsize' must be a positive number"):
            heatmap(title='Title', df=sample_df, fontsize=0)


## bin_and_plot tests

class TestBinAndPlot:
    @staticmethod
    def test_bin_and_plot_returns_correct_data(sample_df):
        """Test if the new binned column is created correctly and returned."""
        bins = [0, 2.5, 5.0]

        df_result = bin_and_plot(title='Title', label='Label', df=sample_df, col='A', new_col='new_A', bins=bins, show_plot=False)

        assert 'new_A' in df_result.columns
        assert isinstance(df_result['new_A'].dtype, pd.CategoricalDtype)

        counts = df_result['new_A'].value_counts()
        assert len(counts) == 2
        assert sorted(counts.to_list()) == [2, 2]
        assert df_result.shape[0] == sample_df.shape[0]
        assert isinstance(counts.index[0], pd.Interval)


    @staticmethod
    def test_bin_and_plot_custom_labels_and_order(sample_df):
        """Test correct binning with custom labels, verifying the internal 'plot_order' logic."""
        bins = [0, 2, 4, 6]
        labels = ['Low', 'Medium', 'High']

        df_result = bin_and_plot(title='Title', label='Label', df=sample_df, col='A', new_col='new_A', bins=bins,
                                 labels=labels, show_plot=False)
        
        assert df_result['new_A'].cat.categories.tolist() == labels

        counts = df_result['new_A'].value_counts().drop(labels=['nan'], errors='ignore')
        assert counts.to_dict() == {'Low': 2, 'Medium': 1, 'High': 1}


    @staticmethod
    def test_bin_and_plot_calls_count_plot_with_correct_map(sample_df, mock_count_plot_dependency):
        """Test that `count_plot()` is called with correct arguments, including the binned column and 'plot_order'."""
        mock_count_plot = mock_count_plot_dependency
        bins = 3

        bin_and_plot(title='Title', label='Label', df=sample_df, col='A', new_col='new_A', bins=bins, hue='B',
                     axis='y', palette='Set1', tick_rotation=45, show_plot=True)
        
        mock_count_plot.assert_called_once()
        called_kwargs = mock_count_plot.call_args[1]

        assert called_kwargs['title'] == 'Title'
        assert called_kwargs['label'] == 'Label'
        assert called_kwargs['col'] == 'new_A'
        assert called_kwargs['axis'] == 'y'
        assert called_kwargs['hue'] == 'B'
        assert called_kwargs['palette'] == 'Set1'
        assert called_kwargs['tick_rotation'] == 45
        assert isinstance(called_kwargs['order'], list)
        assert len(called_kwargs['order']) == bins


    @staticmethod
    def test_bin_and_plot_calls_count_plot_with_labels_order(sample_df, mock_count_plot_dependency):
        """Test that custom labels are passed as 'order' to `count_plot()"""
        mock_count_plot = mock_count_plot_dependency
        labels = ['L', 'M', 'H']

        bin_and_plot(title='Title', label='Labels', df=sample_df, col='A', new_col='new_A', bins=3, labels=labels, show_plot=True)

        mock_count_plot.assert_called_once()
        called_kwargs = mock_count_plot.call_args[1]

        assert called_kwargs['order'] == labels


    @staticmethod
    def test_bin_and_plot_skips_plotting(sample_df, mock_count_plot_dependency):
        """Test that `count_plot()` is NOT called when 'show_plot' is False."""
        mock_count_plot = mock_count_plot_dependency

        df_result = bin_and_plot(title='Title', label='Labels', df=sample_df, col='A', new_col='new_A', bins=5, show_plot=False)

        mock_count_plot.assert_not_called()
        assert 'new_A' in df_result.columns


    @staticmethod
    def test_bin_and_plot_value_error_non_numeric_cols(sample_df):
        """Test `bin_and_plot()` raises a RuntimeError when the source column 'col'is non-numeric."""
        with pytest.raises(RuntimeError, match="An unexpected error occurred during binning of column B"):
            bin_and_plot(title='Title', label='Labels', df=sample_df, col='B', new_col='new_B', bins=10)


    @staticmethod
    def test_bin_and_plot_key_error_col(sample_df):
        """Test `bin_and_plot()` raises a KeyError when 'col'is missing."""
        with pytest.raises(KeyError, match="Column 'Missing' not found in DataFrame"):
            bin_and_plot(title='Title', label='Labels', df=sample_df, col='Missing', new_col='new', bins=10)

    
    @staticmethod
    def test_bin_and_plot_type_error_new_col(sample_df):
        """Test `bin_and_plot()` raises a TypeError when 'new_col' is non-string."""
        with pytest.raises(TypeError, match="'title', 'label', 'col', and 'new_col' must be strings"):
            bin_and_plot(title='Title', label='Labels', df=sample_df, col='A', new_col=123, bins=10)


## chi_squared_test test

class TestChiSquaredTest:
    @staticmethod
    def test_chi_squared_test_significant_result(sample_df_chi2, mock_chi2_independence, capfd_out):
        """Test case where H0 is rejected (p <= alpha) with strong association."""
        expected, observed, stats = chi_squared_test(df=sample_df_chi2, col1='B', col2='C', alpha=0.05)

        assert expected is not None
        assert observed is not None
        assert stats is not None
        assert isinstance(stats, pd.DataFrame)

        out, _ = capfd_out.readouterr()

        assert "Reject Null Hypothesis (H0)" in out
        assert "statistically significant association" in out
        assert "Moderate Association" in out

        assert "------Expected Frequencies------" in out
        assert "------Observed Frequencies------" in out
        assert "------Test Statistics Summary------" in out


    @staticmethod
    def test_chi_squared_test_non_significant_result(sample_df_chi2, mock_chi2_independence, capfd_out):
        """Test case where H0 is failed to be rejected (p > alpha)"""
        expected, observed, _ = chi_squared_test(df=sample_df_chi2, col1='B', col2='A', alpha=0.01)

        assert expected is not None
        assert observed is not None

        out, _ = capfd_out.readouterr()

        assert "Fail to Reject the Null Hypothesis (H0)" in out
        assert "no statistically significance evidence of an association" in out
        assert "p-value is 0.5000" in out


    @staticmethod
    def test_chi_squared_test_custom_alpha(sample_df_chi2, mock_chi2_independence, capfd_out):
        """Test using a different alpha to veriy it's used in the conclusion."""
        chi_squared_test(df=sample_df_chi2, col1='B', col2='C', alpha=0.01)

        out, _ = capfd_out.readouterr()

        assert "alpha = 0.01" in out
        assert "Reject Null Hypothesis (H0)" in out


    @staticmethod
    def test_chi_squared_test_assumption_violation(sample_df_chi2, mock_chi2_independence, capfd_out):
        """Test case where the expected cell frequency assumption is violated."""
        expected_viol_data = pd.DataFrame({'x': [1, 1], 'y': [8, 1]}, index=['P', 'Q'])
        stats_sig_data = pd.DataFrame({ 'test': ['pearson'], 'chisq': [12.8], 'dof': [1], 'pval': [0.0003], 'cramer': [0.35], 'power': [0.95]})
        
        def mock_pingouin_violation(data, x, y):
            return expected_viol_data, expected_viol_data, stats_sig_data
        
        with mock.patch('pingouin.chi2_independence', side_effect=mock_pingouin_violation):
            expected, observed, stats = chi_squared_test(df=sample_df_chi2, col1='B', col2='D', alpha=0.05)
        
        assert expected is None
        assert observed is None
        assert stats is None

        out, _ = capfd_out.readouterr()

        assert "Assumption Not Met: 75.0%" in out
        assert "Test aborted." in out


    @staticmethod
    def test_chi_squared_test_strongest_association(sample_df_chi2, monkeypatch, capfd_out):
        """Test the 'very Strong Association' logic (Cramer's V > 0.50)."""
        expected_strong = pd.DataFrame({'x': [5.0, 5.0], 'y': [5.0, 5.0]}, index=['P', 'Q'])
        stats_strong = pd.DataFrame({'test': ['pearson'], 'chisq': [20.0], 'dof': [1], 'pval': [0.00001], 'cramer': [0.85], 'power': [1.0]})
        
        def mock_pingouin_strong(data, x, y):            
            return expected_strong, expected_strong, stats_strong
        
        monkeypatch.setattr('pingouin.chi2_independence', mock_pingouin_strong)

        chi_squared_test(df=sample_df_chi2, col1='B', col2='A', alpha=0.05)

        out, _ = capfd_out.readouterr()

        assert "Cramer's V effect size is 0.8500" in out
        assert "Very Strong Association" in out


    @staticmethod
    def test_chi_squared_test_type_error_col1(sample_df_chi2):
        """Test `chi_squared_test()` raises TypeError when 'col1' is non-string."""
        with pytest.raises(TypeError, match="'col1' and 'col2' must be strings"):
            chi_squared_test(df=sample_df_chi2, col1=123, col2='B', alpha=0.05)


    @staticmethod
    def test_chi_squared_test_type_error_col2(sample_df_chi2):
        """Test `chi_squared_test()` raises TypeError when 'col2' is non-string."""
        with pytest.raises(TypeError, match="'col1' and 'col2' must be strings"):
            chi_squared_test(df=sample_df_chi2, col1='B', col2=123, alpha=0.05)


    @staticmethod
    def test_chi_squared_test_value_error_alpha(sample_df_chi2):
        """Test `chi_squared_test()` raises ValueError when 'alpha' is outside the (0, 1) range."""
        with pytest.raises(ValueError, match="'alpha' must be a float between 0 and 1"):
            chi_squared_test(df=sample_df_chi2, col1='B', col2='C', alpha=1.0)


## generate_data tests

@pytest.fixture(scope='class')
def sample_data():
    return generate_data(n_records=10, seed=42)

class TestGenerateData:
    @staticmethod
    def test_generate_data_output_type_and_shape(sample_data):
        """Test that the output is a DataFrame, has the correct number of columns, and the requested number of rows."""
        expected_cols = 32
        expected_rows = 10

        assert isinstance(sample_data, pd.DataFrame)
        assert sample_data.shape == (expected_rows, expected_cols)


    @staticmethod
    def test_generate_data_columns_names_and_order(sample_data):
        """Test that all required columns are present in the correct order."""
        expected_order = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long',
        'Latitude', 'Longitude', 'Gender', 'Senior Citizen', 'Partner',
        'Dependents', 'Tenure Months', 'Phone Service', 'Multiple Lines',
        'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract',
        'Paperless Billing', 'Payment Method', 'Monthly Charges', 'Total Charges',
        'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason'
        ]

        assert len(sample_data.columns) == len(expected_order)
        assert list(sample_data.columns) == expected_order


    @staticmethod
    def test_generate_data_reproducibility():
        """Test that two DataFrames generated with the same seed are identical."""
        df1 = generate_data(n_records=50, seed=100)
        df2 = generate_data(n_records=50, seed=100)

        pd.testing.assert_frame_equal(df1, df2)


    @staticmethod
    def test_generate_data_id_uniqueness_and_format(sample_data):
        """Test CustomerID uniqueness and format."""
        assert sample_data['CustomerID'].nunique() == sample_data.shape[0]
        assert sample_data['CustomerID'].iloc[0] == '0001-CUSTM'
        assert sample_data['CustomerID'].iloc[-1] == '0010-CUSTM'


    @staticmethod
    def test_generate_data_churn_value_range(sample_data):
        """Test that 'Churn Value' is stricly binary (0 or 1)."""
        assert set(sample_data['Churn Value'].unique()).issubset({0, 1})


    @staticmethod
    def test_generate_data_charges_non_negative(sample_data):
        """Test that all financial columns are non-negative."""
        assert (sample_data['Monthly Charges'] >= 0).all()
        assert (sample_data['Total Charges'] >= 0).all()
        assert (sample_data['CLTV'] >= 0).all()


    @staticmethod
    def test_generate_data_tenure_range(sample_data):
        """Test that 'Tenure Months' is within the expected 1-72 month range."""
        assert sample_data['Tenure Months'].min() >= 1
        assert sample_data['Tenure Months'].max() <= 72


    @staticmethod
    def test_generate_data_multiple_lines_dependency(sample_data):
        """
        Test that if 'Phone Service' is 'No', then 'Multiple Lines' must be 'No phone service'.
        Dependency: 'Multiple Lines' requires 'Phone Service'.
        """
        no_phone_service = sample_data[sample_data['Phone Service'] == 'No']

        assert (no_phone_service['Multiple Lines'] == 'No phone service').all()


    @staticmethod
    def test_generate_data_internet_service_addons_dependency(sample_data):
        """Test that if 'Internet Service' is 'No', then all internet-dependents add-ons must be 'No internet Service'."""
        no_internet_service = sample_data[sample_data['Internet Service'] == 'No']
        internet_dependent_cols = ['Online Security', 'Online Backup', 'Device Protection',
                               'Tech Support', 'Streaming TV', 'Streaming Movies']
        

        for col in internet_dependent_cols:
            assert (no_internet_service[col] == 'No internet service').all()


    @staticmethod
    def test_generate_data_total_charges_logic(sample_data):
        """Test the crucial logic that for customers with 1 month tenure, 'Total Charges' must be equal to 'Monthly Charges'."""
        new_customers = sample_data[sample_data['Tenure Months'] <= 1]

        if not new_customers.empty:
            np.testing.assert_allclose(new_customers['Total Charges'], new_customers['Monthly Charges'], atol=0.01)


    @staticmethod
    def test_generate_data_churn_reason_dependency(sample_data):
        """Test that 'Churn Reason' is only present (not empty string) for Churned Customers."""
        non_churned = sample_data[sample_data['Churn Value'] == 0]

        assert (non_churned['Churn Reason'] == '').all()