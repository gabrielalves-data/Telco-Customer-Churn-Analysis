import pytest
import pandas as pd
import numpy as np
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