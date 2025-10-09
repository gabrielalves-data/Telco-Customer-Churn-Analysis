import pytest
import pandas as pd
import numpy as np
from unittest import mock
import os

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


## safe_display tests

class TestSafeDisplay:
    @staticmethod
    def test_safe_display_uses_display_when_available(sample_df):
        """Test that safe_display uses IPython.display.display when available."""
        mock_display = mock.MagicMock()
        with mock.patch.dict('sys.modules', {'IPython.display': mock.MagicMock(display=mock_display)}):
            result = safe_display(sample_df)
        
        mock_display.assert_called_once_with(sample_df)
        assert result.equals(sample_df)


    @staticmethod
    @mock.patch('builtins.print')
    def test_safe_display_falls_back_to_print(mock_print):
        """Test that safe_display falls back to print when display is not available."""
        data = 'Test string'
        with mock.patch.dict('builtins.__dict__', {'display': None}, clear=True):
            safe_display(data)
        
        mock_print.assert_called_once_with(data)


## read_excel tests

class TestReadExcel:
    @staticmethod
    @pytest.fixture
    def mock_excel_file(tmp_path):
        """Creates a temporary dummy Excel file for testing."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': ['2021-01-01', '2021-02-01']})
        file_path = tmp_path / "test.xlsx"
        df.to_excel(file_path, index=False)
        return str(file_path)

    
    @staticmethod
    def test_read_excel_success(mock_excel_file):
        """Test successful reading a valid Excel file without date parsing."""
        df = read_excel(mock_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert df['A'].dtype == 'int64'


    @staticmethod
    def test_read_excel_with_date_parsing(mock_excel_file):
        """Test successful reading an Excel file with date parsing."""
        df = read_excel(mock_excel_file, date_cols=['C'])
        assert df['C'].dtype == 'datetime64[ns]'


    @staticmethod
    def test_read_excel_raises_file_not_found_error():
        """Test that read_excel raises FileNotFoundError for a non-existent file."""
        with pytest.raises(FileNotFoundError, match='Error: File not found'):
            read_excel("non_existent_file.xlsx")


    @staticmethod
    @mock.patch('pandas.read_excel')
    def test_read_excel_raises_import_error(mock_read_excel):
        """Test that read_excel raises ImportError when no engine is available."""
        mock_read_excel.side_effect = ImportError("No engine 'openpyxl' or 'xldr' found")
        with pytest.raises(ImportError, match="Missing Excel engine"):
            read_excel("mock_path.xlsx")


    @staticmethod
    @mock.patch('pandas.read_excel')
    @mock.patch('builtins.print')
    def test_read_excel_warns_on_empty_df(mock_print, mock_read_excel):
        """Test that read_excel warns when an empty DataFrame is returned."""
        mock_read_excel.return_value = pd.DataFrame()
        df = read_excel("mock_path.xlsx")
        mock_print.assert_called_once_with(mock.ANY)
        assert df.empty


## df_head tests

class TestDfHead:
    
    @staticmethod
    def test_df_head_returns_correct_tuple(sample_df):
        """Test that df_head returns a tuple of (head, describe) DataFrames with correct content."""
        with mock.patch('src.telco_customer_churn_analysis.utils.safe_display', side_effect=lambda x: x) as mock_safe_display:
            head, describe = df_head(sample_df)

        assert isinstance(head, pd.DataFrame)
        assert isinstance(describe, pd.DataFrame)
        assert head.shape == (5, 4)


    @staticmethod
    def test_df_head_with_custom_n_rows(sample_df):
        """Test that df_head returns correct number of rows when n_rows is specified."""
        with mock.patch('src.telco_customer_churn_analysis.utils.safe_display', side_effect=lambda x: x):
            head, _ = df_head(sample_df, n_rows=3)
        assert head.shape == (3, 4)


    @staticmethod
    def test_df_head_raises_type_error():
        """Test that df_head raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            df_head([1, 2, 3])


    @staticmethod
    @mock.patch('builtins.print')
    def test_df_head_invalid_n_rows_warning(mock_print, sample_df):
        """Test that df_head warns when n_rows is invalid."""
        with mock.patch('src.telco_customer_churn_analysis.utils.safe_display', side_effect=lambda x: x):
            df_head(sample_df, n_rows=-1)
        mock_print.assert_called_once_with("Warning: n_rows should be a non-negative integer. Using default value 5.")


## col_replace tests

class TestColReplace:
    @staticmethod
    def test_col_replace_single_value(sample_df):
        """Test replacing a single value in a column."""
        new_df = col_replace(sample_df, 'A', 2, 20)
        pd.testing.assert_series_equal(new_df['A'].values, np.array([1, 20, np.nan, 4, 5]))
        assert sample_df['A'].iloc[1] == 2


    @staticmethod
    def test_col_replace_list_of_values(sample_df):
        """Test replacing a list of values in a column."""
        new_df = col_replace(sample_df, 'B', ['x', 'y'], 'replaced')
        assert (new_df['B'].values == np.array(['replaced', 'replaced', 'z', 'replaced', 'replaced'])).all()


    @staticmethod
    def test_col_replace_warns_on_no_occurrence_single(mock_print, sample_df):
        """Test that col_replace warns when no occurrences of the value to replace are found (single value)."""
        col_replace(sample_df, 'A', 99, 100)
        mock_print.assert_called_once_with("Warning: No occurrences of 99 found in column 'A'.")


    @staticmethod
    def test_col_replace_raises_type_error():
        """Test that col_replace raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            col_replace('not_df', 'col', 100, 200)


    @staticmethod
    def test_col_replace_raises_key_error(sample_df):
        """Test that col_replace raises KeyError when column does not exist."""
        with pytest.raises(KeyError, match="Column 'Z' not found in DataFrame."):
            col_replace(sample_df, 'Z', 1, 10)


## null_rows tests

class TestNullRows:
    @staticmethod
    def test_null_rows_all_columns(sample_df):
        """Test null_rows function for all columns."""
        result = null_rows(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_df.shape
        assert result['A'].sum() == 1


    @staticmethod
    def test_null_rows_single_column(sample_df):
        """Test null_rows function for a single specified column."""
        result = null_rows(sample_df, columns=['A'])
        assert isinstance(result, pd.Series)
        assert result.sum() == 1


    @staticmethod
    def test_null_rows_multiple_columns(sample_df):
        """Test null_rows function for multiple specified columns."""
        result = null_rows(sample_df, columns=['A', 'C'])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'C']
        assert result['A'].sum() == 1


    @staticmethod
    def test_null_rows_raises_type_error():
        """Test that null_rows raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            null_rows('not_df')


    @staticmethod
    def test_null_rows_raises_key_error(sample_df):
        """Test that null_rows raises KeyError when specified columns do not exist."""
        with pytest.raises(KeyError, match="Column 'Z' not found in DataFrame."):
            null_rows(sample_df, columns=['Z'])


## df_loc tests

class TestDfLoc:
    @staticmethod
    def test_df_loc_single_condition(sample_df):
        """Test df_loc with a single condition."""
        condition = (sample_df['A'] > 1)
        result = df_loc(sample_df, condition, ['A', 'B'])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        assert result.shape == (3, 2)


    @staticmethod
    def test_df_loc_multiple_conditions(sample_df):
        """Test df_loc with multiple conditions."""
        conditions = sample_df['B'] == 'y'
        result = df_loc(sample_df, conditions, ['A', 'B'])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['A', 'B']
        assert result.shape == (2, 2)


    @staticmethod
    def test_df_loc_raises_type_error_df():
        """Test that df_loc raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            df_loc('not_df', [True] * 5, 'A')


    @staticmethod
    def test_df_loc_raises_type_error_col():
        """Test that df_loc raises TypeError when col is not a list of strings or a string."""
        with pytest.raises(TypeError, match="Input 'col' must be a string or a list of strings."):
            df_loc(sample_df, [True] * 5, 123)


    @staticmethod
    def test_df_loc_raises_key_error(sample_df):
        """Test that df_loc raises KeyError when specified columns do not exist."""
        condition = sample_df['A'] > 1
        with pytest.raises(KeyError, match="Columns specified were not found in DataFrame: 'Z'."):
            df_loc(sample_df, condition, ['Z'])


    @staticmethod
    def test_df_loc_raises_index_error(sample_df):
        """Test that df_loc raises IndexError when condition length does not match DataFrame length."""
        condition = [True, False]  # Incorrect length
        with pytest.raises(IndexError, match="Length of 'condition' must match number of rows in DataFrame."):
            df_loc(sample_df, condition, ['A'])


## df_aggfunc tests

class TestDfAggfunc:
    @staticmethod
    def test_df_aggfunc_single_agg_single_column(sample_df):
        """Test df_aggfunc with a single column and single aggregation function."""
        result = df_aggfunc(sample_df, 'mean', 'A')
        assert isinstance(result, (int, float, pd.Series))
        assert np.isclose(result, 3.0)


    @staticmethod
    def test_df_aggfunc_list_df(sample_df):
        """Test df_aggfunc with a list of aggregation functions the entire numeric DataFrame."""
        result = df_aggfunc(sample_df, ['mean', 'sum'], ['A', 'C'])
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        assert result.loc['A', 'mean'] == 3.0
        assert result.loc['A', 'sum'] == 12.0


    @staticmethod
    def test_df_aggfunc_dict_df(sample_df):
        """Test df_aggfunc with a dictionary of aggregation functions for specific columns."""
        result = df_aggfunc(sample_df, {'A': 'mean', 'C': 'sum'})
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)
        assert result['A'].loc[0] == 3.0
        assert result['C'].loc['sum'] == 11.0


    @staticmethod
    def test_df_aggfunc_value_counts_single_column(sample_df):
        """Test df_aggfunc with 'value_counts' for a single column."""
        result = df_aggfunc(sample_df, 'value_counts', 'B')
        assert isinstance(result, pd.Series)
        assert result.shape == (3, 1)
        assert result.loc['x'] == 2
        assert result.loc['y'] == 2
        assert result.loc['z'] == 1
        assert result.name == 'B'


    @staticmethod
    def test_df_aggfunc_value_counts_list_columns(sample_df):
        """Test df_aggfunc with 'value_counts' for a list of columns."""
        result = df_aggfunc(sample_df, 'value_counts', ['B', 'A'])
        assert isinstance(result, dict)
        assert 'B' in result and 'A' in result
        assert result['B'].loc['x'] == 2
        assert result['A'].loc[1] == 1


    @staticmethod
    def test_df_aggfunc_custom_callable(sample_df):
        """Test df_aggfunc with a custom callable function (lambda)."""
        custom_median = lambda x: np.nanmedian(x) * 2
        result = df_aggfunc(sample_df, custom_median, ['A', 'C'])
        assert isinstance(result, pd.Series)
        assert result['A'].loc[0] == 6.0
        assert result['C'].loc[0] == 6.6


    @staticmethod
    def test_df_aggfunc_raises_type_error_df():
        """Test that df_aggfunc raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            df_aggfunc('not_df', 'mean', 'A')


    @staticmethod
    def test_df_aggfunc_key_error_column(sample_df):
        """Test that df_aggfunc raises KeyError when specified columns do not exist."""
        with pytest.raises(KeyError, match="Columns specified were not found in DataFrame: 'Z'."):
            df_aggfunc(sample_df, 'mean', ['Z'])


    @staticmethod
    def test_df_aggfunc_value_error_invalid_agg(sample_df):
        """Test that df_aggfunc raises ValueError when aggregation function is incompatile with data type."""
        with pytest.raises(ValueError, match="The aggregation function 'mean' is not available or compatible."):
            df_aggfunc(sample_df, 'mean', 'B')


    @staticmethod
    def test_df_aggfunc_value_error_value_counts_without_column(sample_df):
        """Test that df_aggfunc raises ValueError when 'value_counts' is used without specifying columns."""
        with pytest.raises(ValueError, match="When using 'value_counts', 'col' must be specified as a string or list of strings."):
            df_aggfunc(sample_df, 'value_counts')


    @staticmethod
    def test_df_aggfunc_type_error_invalid_column_type(sample_df):
        """Test that df_aggfunc raises TypeError when col is not a string, list, or None."""
        with pytest.raises(TypeError, match="Input 'col' must be a string, list of strings, or None."):
            df_aggfunc(sample_df, 'mean', 123)


    @staticmethod
    def test_df_aggfunc_empty_df(empty_df):
        """Test that df_aggfunc handles an empty DataFrame."""
        result = df_aggfunc(empty_df, 'mean', ['A', 'B'])
        assert isinstance(result, pd.Series)
        assert len(result) == 2


## drop_labels tests

class TestDropLabels:
    @staticmethod
    def test_drop_labels_single_column(sample_df):
        """Test drop_labels with a single column."""
        new_df = drop_labels(sample_df, 'A', 1)
        assert 'A' in new_df.columns
        assert sample_df.shape == (5, 4)
        pd.testing.assert_index_equal(new_df.index, pd.Index(['B', 'C', 'D']))


    @staticmethod
    def test_drop_labels_list_of_columns(sample_df):
        """Test drop_labels with a list of columns."""
        new_df = drop_labels(sample_df, ['B', 'C'], 'columns')
        assert 'B' not in new_df.columns
        assert 'C' not in new_df.columns
        assert sample_df.shape == (5, 4)
        assert new_df.shape == (5, 2)


    @staticmethod
    def test_drop_labels_single_row(sample_df):
        """Test drop_labels with a single row index."""
        new_df = drop_labels(sample_df, 0, 'index')
        assert 0 not in new_df.index
        assert new_df.shape == (4, 4)


    @staticmethod
    def test_drop_labels_list_of_rows(sample_df):
        """Test drop_labels with a list of row indices."""
        new_df = drop_labels(sample_df, [0, 1], 'index')
        assert 0 not in new_df.index
        assert 1 not in new_df.index
        assert new_df.shape == (3, 4)


    @staticmethod
    def test_drop_labels_ignore_non_existent_labels(sample_df):
        """Test drop_labels that non-existent labels are ignored."""
        new_df = drop_labels(sample_df, ['Z', 'B'], 'columns')
        assert 'B' not in new_df.columns
        assert new_df.shape == (5, 3)


    @staticmethod
    def test_drop_labels_no_label_return_copy(sample_df):
        """Test drop_labels returns a copy of the original DataFrame when no labels are provided."""
        new_df = drop_labels(sample_df, None, 'columns')
        pd.testing.assert_frame_equal(new_df, sample_df)
        assert new_df is not sample_df


    @staticmethod
    def test_drop_labels_raises_type_error_df():
        """Test that drop_labels raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            drop_labels('not_df', 'A', 'columns')


    @staticmethod
    def test_drop_labels_raises_type_error_labels(sample_df):
        """Test that drop_labels raises TypeError when labels is not a string, list, int, or None."""
        with pytest.raises(TypeError, match="Input 'labels' must be a string, list of strings, int, list of ints, or None."):
            drop_labels(sample_df, 12.34, 'columns')


    @staticmethod
    def test_drop_labels_raises_type_error_axis(sample_df):
        """Test that drop_labels raises TypeError when axis is an invalid type."""
        with pytest.raises(ValueError, match="Input 'axis' must be either (0 or 1) or ('columns' or 'index')."):
            drop_labels(sample_df, 'A', 'invalid_axis')


    @staticmethod
    def test_drop_labels_raises_value_error_axis(sample_df):
        """Test that drop_labels raises ValueError when axis is not 0, 1, 'columns', or 'index'."""
        with pytest.raises(ValueError, match="Input 'axis' must be either (0 or 1) or ('columns' or 'index')."):
            drop_labels(sample_df, 'A', 2)


    @staticmethod
    def test_drop_labels_empty_df(empty_df):
        """Test that drop_labels handles an empty DataFrame."""
        new_df = drop_labels(empty_df, 'A', 'columns')
        pd.testing.assert_frame_equal(new_df, pd.DataFrame())


## Visualization function tests (count_plot, histogram, heatmap)

## count_plot tests

@pytest.fixture
def numeric_df():
    """Provides a sample DataFrame with numeric data for visualization tests."""
    data = {
        'A': [1, 2, 1, 3, 2, 1, 3, 3, 2, 1],
        'B': [10, 20, 10, 30, 20, 10, 30, 30, 20, 10],
        'C': [1.1, 2.2, 1.1, 3.3, 2.2, 1.1, 3.3, 3.3, 2.2, 1.1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def categorical_df():
    """Provides a sample DataFrame with categorical data for visualization tests."""
    data = {
        'A': ['cat', 'dog', 'cat', 'mouse', 'dog', 'cat', 'mouse', 'mouse', 'dog', 'cat'],
        'B': ['red', 'blue', 'red', 'green', 'blue', 'red', 'green', 'green', 'blue', 'red'],
        'C': ['type1', 'type2', 'type1', 'type3', 'type2', 'type1', 'type3', 'type3', 'type2', 'type1'],
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def mock_plotting_libraries():
    plt_mock = mock.MagicMock()
    ax_mock = mock.MagicMock()
    ax_mock.containers = [mock.MagicMock()]
    plt_mock.subplots.return_value = (mock.MagicMock(), ax_mock)

    sns_mock = mock.MagicMock()

    with mock.patch('matplotlib.pyplot', plt_mock), mock.patch('seaborn', sns_mock):
        yield plt_mock, sns_mock


class TestCountPlot:
    @staticmethod
    def test_count_plot_successful_vertical(sample_df,mock_plotting_libraries):
        """Test successful creation of a vertical count plot."""
        plt_mock, sns_mock = mock_plotting_libraries
        
        count_plot('Test title', 'Test label', sample_df, 'B')

        sns_mock.countplot.assert_called_once()
        call_kwargs = sns_mock.countplot.call_args[1]
        assert call_kwargs['x'] == 'B'
        assert call_kwargs['data'].equals(sample_df)
        assert call_kwargs['hue'] is None

        plt_mock.subplots.assert_called_once()
        ax_mock = plt_mock.subplots.return_value[1]
        ax_mock.set_title.assert_called_once_with('Test title')
        ax_mock.set_xlabel.assert_called_once_with('Test label')
        ax_mock.set_ylabel.assert_called_once_with('Count')

        plt_mock.tight_layout.assert_called_once()
        plt_mock.show.assert_called_once()


    @staticmethod
    def test_count_plot_successful_horizontal_hue(sample_df,mock_plotting_libraries):
        """Test successful creation of a horizontal count plot, with 'hue' and 'tick_rotation' variables."""
        plt_mock, sns_mock = mock_plotting_libraries
        
        count_plot('Test title', 'Test label', sample_df, 'B', axis='y', hue='A', tick_rotation=45)

        sns_mock.countplot.assert_called_once()
        call_kwargs = sns_mock.countplot.call_args[1]
        assert call_kwargs['y'] == 'B'
        assert call_kwargs['hue'] == 'A'

        ax_mock = plt_mock.subplots.return_value[1]
        ax_mock.set_title.assert_called_once_with('Test title')
        ax_mock.set_ylabel.assert_called_once_with('Test label')
        ax_mock.set_xlabel.assert_called_once_with('Count')
        ax_mock.set_yticklabels.assert_called_once()
        assert ax_mock.set_yticklabels.call_args[1]['rotation'] == 45


    @staticmethod
    @pytest.mark.parametrize("invalid_input", ['invalid', 2, None, [], {}])
    def test_count_plot_raises_type_error_df(invalid_input):
        """Test that count_plot raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            count_plot('Title', 'Label', invalid_input, 'A')


    @staticmethod
    @pytest.mark.parametrize("invalid_col", ['Z', 'NonExistent', 'Existent', 'Non'])
    def test_count_plot_raises_key_error_col(sample_df, invalid_col):
        """Test that count_plot raises KeyError when column does not exist."""
        with pytest.raises(KeyError, match=f"Column '{invalid_col}' not found in DataFrame."):
            count_plot('Title', 'Label', sample_df, invalid_col)

    
    @staticmethod
    def test_count_plot_raises_key_error_hue(sample_df):
        """Test that count_plot raises KeyError when hue column does not exist."""
        with pytest.raises(KeyError, match="Hue column 'Z' not found in DataFrame."):
            count_plot('Title', 'Label', sample_df, 'B', hue='Z')


    @staticmethod
    @pytest.mark.parametrize("invalid_axis", ['invalid', 2, 3.5])
    def test_count_plot_raises_value_error_axis(sample_df, invalid_axis):
        """Test that count_plot raises ValueError when axis is not 'x' or 'y'."""
        with pytest.raises(ValueError, match="Input 'axis' must be either 'x' or 'y'."):
            count_plot('Title', 'Label', sample_df, 'B', axis=invalid_axis)


## histogram tests
class TestHistogram:
    @staticmethod
    def test_histogram_successful_vertical_with_kde(numeric_df,mock_plotting_libraries):
        """Test successful creation of a vertical histogram with KDE."""
        plt_mock, sns_mock = mock_plotting_libraries
        
        histogram('Histogram Title', 'X Label', numeric_df, 'A', bins=10, kde=True)

        sns_mock.histplot.assert_called_once()
        call_kwargs = sns_mock.histplot.call_args[1]
        assert call_kwargs['x'] == 'A'
        assert call_kwargs['data'].equals(numeric_df)
        assert call_kwargs['bins'] == 10
        assert call_kwargs['kde'] is True

        plt_mock.xlabel.assert_called_once_with('X Label')
        ax_mock = plt_mock.subplots.return_value[1]
        ax_mock.set_title.assert_called_once_with('Histogram Title')
        ax_mock.set_xlabel.assert_called_once_with('X Label')

        plt_mock.tight_layout.assert_called_once()
        plt_mock.show.assert_called_once()


    @staticmethod
    def test_histogram_successful_horizontal_hue(categorical_df,mock_plotting_libraries):
        """Test successful creation of a horizontal histogram with hue."""
        plt_mock, sns_mock = mock_plotting_libraries
        
        df_merged = categorical_df

        histogram('Histogram Title', 'Y Label', df_merged, 'A', axis='y', bins=10, hue='B')

        sns_mock.histplot.assert_called_once()
        call_kwargs = sns_mock.histplot.call_args[1]
        assert call_kwargs['y'] == 'A'
        assert call_kwargs['hue'] == 'B'
        assert call_kwargs['bins'] == 10

        plt_mock.ylabel.assert_called_once_with('Y Label')
        ax_mock = plt_mock.subplots.return_value[1]
        ax_mock.set_title.assert_called_once_with('Histogram Title')


    @staticmethod
    @pytest.mark.parametrize("invalid_input", ['invalid', 2, None, [], {}])
    def test_histogram_raises_type_error_df(invalid_input):
        """Test that histogram raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            histogram('Title', 'Label', invalid_input, 'A')


    @staticmethod
    def test_histogram_raises_key_error_col(numeric_df):
        """Test that histogram raises KeyError when column does not exist."""
        with pytest.raises(KeyError, match="Column 'Z' not found in DataFrame."):
            histogram('Title', 'Label', numeric_df, 'Z')


    @staticmethod
    def test_histogram_raises_runtime_error_non_numeric_col(categorical_df):
        """Test that histogram raises RuntimeError when column is non-numeric."""
        plt_mock, sns_mock = mock_plotting_libraries
        sns_mock.histplot.side_effect = Exception("RuntimeError: An unexpected error occurred during plot generation. Check if column 'col' is numerical.")
        with pytest.raises(RuntimeError, match="RuntimeError: An unexpected error occurred during plot generation."):
            histogram('Title', 'Label', categorical_df, 'A')


## heatmap tests
class TestHeatmap:
    @staticmethod
    def test_heatmap_successful(numeric_df,mock_plotting_libraries):
        """Test successful creation of a heatmap."""
        plt_mock, sns_mock = mock_plotting_libraries
        
        heatmap('Heatmap Title', numeric_df)

        sns_mock.heatmap.assert_called_once()
        call_kwargs = sns_mock.heatmap.call_args[1]

        assert isinstance(sns_mock.heatmap.call_args[0][0], pd.DataFrame)
        assert list(sns_mock.heatmap.call_args[0][0].columns) == ['A', 'B', 'C']

        assert call_kwargs['data'].shape == (3, 3)
        assert call_kwargs['annot'] is True
        assert call_kwargs['fmt'] == '.2f'
        assert call_kwargs['cmap'] == 'coolwarm'
        assert call_kwargs['annot_kws']['fontsize'] == 7


        plt_mock.title.assert_called_once_with('Heatmap Title')
        plt_mock.xticks.assert_called_once()
        plt_mock.yticks.assert_called_once()
        plt_mock.tight_layout.assert_called_once()
        plt_mock.show.assert_called_once()


    @staticmethod
    def test_heatmap_custom_params(numeric_df,mock_plotting_libraries):
        """Test successful creation of a heatmap with custom parameters."""
        plt_mock, sns_mock = mock_plotting_libraries
        
        heatmap('Custom Heatmap', numeric_df, annot=False, fontsize=10 ,cmap='viridis', num_decimals=3)

        sns_mock.heatmap.assert_called_once()
        call_kwargs = sns_mock.heatmap.call_args[1]

        assert call_kwargs['annot'] is False
        assert call_kwargs['cmap'] == 'viridis'
        assert call_kwargs['annot_kws']['fontsize'] == 10
        assert call_kwargs['fmt'] == '.3f'


    @staticmethod
    def test_heatmap_raises_type_error_df():
        """Test that heatmap raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            heatmap('Title', 'not_df')


    @staticmethod
    def test_heatmap_raises_value_error_no_numeric_columns(categorical_df):
        """Test that heatmap raises ValueError when DataFrame has no numeric columns."""
        with pytest.raises(ValueError, match="DataFrame contains no numeric columns to calculate a correlation matrix."):
            heatmap('Title', categorical_df)


    @staticmethod
    @pytest.mark.parametrize("invalid_fontsize", ['invalid', -1, 0, None, []])
    def test_heatmap_raises_value_error_invalid_fontsize(numeric_df, invalid_fontsize):
        """Test that heatmap raises ValueError when fontsize is invalid."""
        with pytest.raises(ValueError, match="Input 'fontsize' must be a positive number."):
            heatmap('Title', numeric_df, fontsize=invalid_fontsize)


    @staticmethod
    def test_heatmap_raises_value_error_invalid_num_decimals(numeric_df):
        """Test that heatmap raises ValueError when num_decimals is invalid."""
        with pytest.raises(ValueError, match="Input 'num_decimals' must be a non-negative integer."):
            heatmap('Title', numeric_df, num_decimals=-1)


## bin and plot tests
class TestBinAndPlot:
    @staticmethod
    def test_bin_and_plot_success_binning_only(sample_df):
        """Test successful binning when show_plot=False."""
        df_new = bin_and_plot('Title', 'Label', sample_df, col='A', new_col='A_binned', bins=3, show_plot=False)

        assert isinstance(df_new, pd.DataFrame)
        assert 'A_binned' in df_new.columns
        assert df_new.shape == (5, 5)
        pd.testing.assert_frame_equal(df_new.drop(columns=['A_binned']), sample_df)

        assert df_new['A_binned'].loc[0] == pd.Interval(0.996, 2.333, closed='right')
        assert df_new['A_binned'].loc[2] is np.nan


    @staticmethod
    def test_bin_and_plot_success_binning_with_plot(sample_df,mock_plotting_libraries):
        """Test successful binning and plotting when show_plot=True."""
        plt_mock, sns_mock = mock_plotting_libraries
        
        with mock.patch('src.telco_customer_churn_analysis.utils.count_plot') as mock_count_plot:
            bin_and_plot('Title', 'Label', sample_df, col='A', new_col='A_binned', bins=3, show_plot=True)


        mock_count_plot.assert_called_once()

        call_kwargs = mock_count_plot.call_args[1]
        assert isinstance(call_kwargs['df'], pd.DataFrame)
        assert call_kwargs['df'].shape == (5, 5)
        assert call_kwargs['col'] == 'A_binned'
        assert call_kwargs['hue'] is None
        assert 'A_binned' in call_kwargs['df'].columns


    @staticmethod
    def test_bin_and_plot_with_custom_labels(sample_df):
        """Test bin_and_plot with custom labels."""
        custom_labels = ['Low', 'Medium', 'High']
        df_new = bin_and_plot('Title', 'Label', sample_df, col='A', new_col='A_binned', bins=3, labels=custom_labels, show_plot=False)

        assert pd.api.types.is_categorical_dtype(df_new['A_binned'])
        assert df_new['A_binned'].cat.categories.tolist() == custom_labels
        assert df_new['A_binned'].loc[0] == 'Low'
        assert df_new['A_binned'].loc[1] == 'Medium'


    @staticmethod
    def test_bin_and_plot_raises_type_error_df():
        """Test that bin_and_plot raises TypeError when input is not a DataFrame."""
        with pytest.raises(TypeError, match="Input 'df' must be a pandas DataFrame."):
            bin_and_plot('Title', 'Label', 'not_df', col='A', new_col='A_binned', bins=3)


    @staticmethod
    def test_bin_and_plot_raises_value_error_no_numeric(sample_df):
        """Test that bin_and_plot raises ValueError when column to bin is non-numeric."""
        with pytest.raises(ValueError, match="Invalid binning configuration."):
            bin_and_plot('Title', 'Label', sample_df, col='B', new_col='B_binned', bins=3)


    @staticmethod
    def test_bin_and_plot_raises_key_error_col(sample_df):
        """Test that bin_and_plot raises KeyError when column to bin does not exist."""
        with pytest.raises(KeyError, match="Column 'Z' not found in DataFrame."):
            bin_and_plot('Title', 'Label', sample_df, col='Z', new_col='Z_binned', bins=3)


## chi_square_test_of_independence tests

@pytest.fixture
def categorical_df_for_chi_square():
    """Provides a sample DataFrame with categorical data for chi-square test."""
    data = {
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Service': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'DSL'],
        'A': [1, 2, 1, 3, 2]
    }

    return pd.DataFrame(data)

@pytest.fixture
def mock_chi2_result_passing():
    """Mocks the pingouin.chi2_independence result for a successful test."""
    observed = pd.DataFrame({
        'Partner': ['Yes', 'No'],
        'Female': [30, 10],
        'Male': [20, 40]
    }).set_index('Partner').T


    expected = pd.DataFrame({
        'Partner': ['Yes', 'No'],
        'Female': [25, 15],
        'Male': [25, 35]
    }).set_index('Partner').T


    stats = pd.DataFrame({
        'test': ['pearson', 'G-test'],
        'dof': [1, 1],
        'chi2': [4.0, 4.5],
        'pval': [0.0655, 0.0519],
        'cramer': [0.2, 0.21],
        'power': [0.70, 0.60]
    })

    return observed, expected, stats
    

@pytest.fixture
def mock_chi2_result_reject_h0():
    """Mocks the pingouin.chi2_independence result for a test that rejects H0."""
    observed = pd.DataFrame({
        'Partner': ['Yes', 'No'],
        'Female': [10, 40],
        'Male': [5, 45]
    }).set_index('Partner').T

    excepted = pd.DataFrame({
        'Partner': ['Yes', 'No'],
        'Female': [20, 30],
        'Male': [15, 35]
    }).set_index('Partner').T

    stats = pd.DataFrame({
        'test': ['pearson', 'G-test'],
        'dof': [1, 1],
        'chi2': [15.0, 16.5],
        'pval': [0.0001, 0.00005],
        'cramer': [0.51, 0.52],
        'power': [0.95, 0.97]
    })

    return observed, excepted, stats
    

@pytest.fixture
def mock_chi2_result_assumption_fail():
    """Mock pingouin result where many expected cells are < 5."""

    expected = pd.DataFrame({
        'Service': ['DSL', 'Fiber optic'],
        'Female': [4, 2],
        'Male': [1, 3]
    }).set_index('Service').T

    observed = expected.copy()

    stats = pd.DataFrame({
        'test': ['pearson', 'G-test'],
        'pval': [0.3, 0.2]
    })

    return observed, expected, stats
    

@pytest.fixture(autouse=True)
def mock_pingouin():
    with mock.patch('pingouin', autospec=True) as mock_pg:
        yield mock_pg


class TestChiSquareTestOfIndependence:
    @staticmethod
    def test_chi_square_success_fail_to_reject(categorical_df_for_chi_square, mock_pingouin, mock_chi2_result_passing, capsys):
        """Test successful execution where the null hypothesis is NOT rejected."""
        mock_pg = mock_pingouin
        mock_pg.chi2_independence.return_value = mock_chi2_result_passing

        expected, observed, stats = chi_squared_test(categorical_df_for_chi_square, 'Gender', 'Partner', alpha=0.05)

        mock_pg.chi2_independence.assert_called_once_with(data=categorical_df_for_chi_square, x='Gender', y='Partner')

        assert expected.equals(mock_chi2_result_passing[0])
        assert observed.equals(mock_chi2_result_passing[1])
        assert stats.equals(mock_chi2_result_passing[2])

        captured = capsys.readouterr()
        assert "Fail to Reject the Null Hypothesis (H0)" in captured.out
        assert "no statistically significance evidence of an association" in captured.out
        assert "The p-val is 0.0655" in captured.out


    @staticmethod
    def test_success_reject_h0(categorical_df_for_chi_square, mock_pingouin, mock_chi2_result_reject_h0, capsys):
        """Test successful execution where the null hypothesis is rejected, and Cramer's V interpreted."""
        mock_pg = mock_pingouin
        mock_pg.chi2_independence.return_value = mock_chi2_result_reject_h0

        expected, observed, stats = chi_squared_test(categorical_df_for_chi_square, 'Gender', 'Partner', alpha=0.05)

        captured = capsys.readouterr()
        assert "Reject the null Hypothesis (H0)" in captured.out
        assert "statistically significant association" in captured.out
        assert "**Very Strong Association**" in captured.out
        assert "Statistical power of the test is 0.95" in captured.out


    @staticmethod
    def test_chi_square_assumption_failure(categorical_df_for_chi_square, mock_pingouin, mock_chi2_result_assumption_fail, capsys):
        """Test the function correctly aborts and returns None if the expected frequency assumption is not met (>20% of cells < 5)."""
        mock_pg = mock_pingouin
        mock_pg.chi2_independence.return_value = mock_chi2_result_assumption_fail

        expected, observed, stats = chi_squared_test(categorical_df_for_chi_square, 'Gender', 'Service')

        assert expected is None
        assert observed is None
        assert stats is None

        captured = capsys.readouterr()

        assert "Assumption Not Met" in captured.out
        assert "50%" in captured.out
        assert "Test aborted." in captured.out


    @staticmethod
    def test_chi_squared_raises_runtime_error(categorical_df_for_chi_square, mock_pingouin):
        """Tests that the function handles unexpected exceptions during the pingouin call."""
        mock_pingouin.chi2_independence.side_effect = Exception("A low memory error occurred.")

        with pytest.raises(RuntimeError, match="An unexpected error occurred during Chi-Squared test"):
            chi_squared_test(categorical_df_for_chi_square, 'Gender', 'Partner')


    @staticmethod
    def test_chi_squared_raises_key_error(categorical_df_for_chi_square):
        """Test KeyError for missing columns."""
        with pytest.raises(KeyError, match="Column 'Z' not found in DataFrame"):
            chi_squared_test(categorical_df_for_chi_square, 'Gender', 'Z')


    @staticmethod
    @pytest.mark.parametrize("invalid_alpha", [0.0, 1.0, 1.5, -0.1, 5, '0.05'])
    def test_chi_squared_raises_value_error_alpha(categorical_df_for_chi_square, invalid_alpha):
        """Test ValueError for invalid alpha (not between 0 and 1 exclusive)."""
        with pytest.raises((ValueError, TypeError), match="'alpha' must be a float between 0 and 1"):
            chi_squared_test(categorical_df_for_chi_square, 'Gender', 'Partner', alpha=invalid_alpha)


## generate data test functions

class TestGenerateData:
    @staticmethod
    def test_generate_data_shape_and_columns():
        """Tests that the generated DataFrame has the correct shape and column names"""
        N = 500
        df = generate_data(n_records=N)

        assert df.shape == (N, 32)

        expected_columns = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long',
        'Latitude', 'Longitude', 'Gender', 'Senior Citizen', 'Partner',
        'Dependents', 'Tenure Months', 'Phone Service', 'Multiple Lines',
        'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract',
        'Paperless Billing', 'Payment Method', 'Monthly Charges', 'Total Charges',
        'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason'
        ]

        assert list(df.columns) == expected_columns


    
    @staticmethod
    def test_generate_data_reproducibility():
        """Test that the same seed produces an identical DataFrame."""
        SEED = 100
        df1 = generate_data(n_records=100, seed=SEED)
        df2 = generate_data(n_records=100, seed=SEED)

        pd.testing.assert_frame_equal(df1, df2)


    
    @staticmethod
    def test_generate_data_dependencies_multiple_lines():
        """Tests the logic: 'Multiple lines' = 'No phone service' if 'Phone Service' == 'No'."""
        df = generate_data(n_records=100)

        no_phone_service = df[df['Phone Service'] == 'No']

        assert (no_phone_service['Multiple Lines'] == 'No phone service').all()


    
    @staticmethod
    def test_generate_data_dependencies_internet_addons():
        """Tests the logic: 'Internet Add-ons' = 'No internet service' if 'Internet Service' = 'No'."""
        df = generate_data(n_records=100)

        no_internet_service = df[df['Internet Service'] == 'No']

        internet_dependent_cols = ['Online Security', 'Online Backup', 'Device Protection',
                                   'Tech Support', 'Streaming TV', 'Streaming Movies']
        
        for col in internet_dependent_cols:
            assert (no_internet_service[col] == 'No internet service').all()


    
    @staticmethod
    def test_generate_data_total_charges_logic():
        """Tests the logic that 'Total Charges' must be equal to 'Monthly Charges' of 'Tenure Months' is 1."""
        df = generate_data(n_records=1000)

        tenure_1_customers = df[df['Tenure Months'] == 1]

        pd.testing.assert_series_equal(tenure_1_customers['Total Charges'],
                                       tenure_1_customers['Monthly Charges'],
                                       check_names=False,
                                       check_exact=False,
                                       atol=0.01)
        

    @staticmethod
    def test_generate_data_churn_reason_logic():
        """Tests the logic: 'Churn Reason' must be empty if 'Churn Value' = 0."""
        df = generate_data(n_records=100)

        non_churned_customers = df[df['Churn Value'] == 0]

        assert (non_churned_customers['Churn Reason'] == '').all()