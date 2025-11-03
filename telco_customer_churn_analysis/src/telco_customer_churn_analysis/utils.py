import pandas as pd
from typing import Union, Dict, List, Any, Optional, Tuple, Callable, Literal
from IPython import display
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pingouin
import warnings
import numpy as np
import random
import os
import kagglehub
from typing import Union, Dict, List, Any, Optional, Tuple, Callable, Literal
import sys

## Data Wrangling Functions

def safe_display(data: Any) -> Any:
    """
    Safely displays data in environments that support 'display()' (like Jupyter/Kaggle),
    or falls back to the standard 'print()' function.

    This prevents NameErrors when running notebook code outside of an
    IPython environment and handles unexpected display failures.

    Parameters
    ----------
    data : Any
        The object to be displayed (e.g., pandas DataFrame, list, dict, etc.).

    Returns
    -------
    Any
        The original data object passed in.

    Raises
    ------
    None
        All caught errors related to display/print are handled internally
        by falling back to print() or printing an error message.
    """

    try:
        display(data)

    except NameError:
        print(data)

    except Exception as e:
        print(f'Error: Failed to display data. Falling back to print(). Details: {e}.', file=sys.stderr)
        print(data)

    return data


def kaggle_download():
    """
    Downloads the Telco Customer Churn dataset from Kaggle or uses a cached file if available.

    This function checks if the dataset file 'Telco_customer_churn.xlsx' exists locally.
    If it does, the local cached file path is returned. Otherwise, it downloads the dataset
    from Kaggle using the `kagglehub.dataset_download` method and returns the full path to the file.

    Parameters
    ----------
    None

    Returns
    -------
    str
        The full file path to the Telco Customer Churn Excel dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file is not found locally and cannot be found after download.
    RuntimeError
        If the Kaggle download fails or returns an invalid path.
    """
    filename = 'Telco_customer_churn.xlsx'

    if os.path.exists(filename):
        full_path = os.path.abspath(filename)
        print(f'Using cached dataset at: {full_path}')

        return full_path

    try:
        path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")
    
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset from Kaggle: {e}")
    
    full_path = os.path.join(path, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset was not found after download at path: {full_path}")
    
    return full_path


def read_excel(path: str, date_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reads a single Excel file into a pandas DataFrame.

    This function is a robust wrapper around pandas.read_excel,
    with optional date parsing and explicit error handling.

    Parameters
    ----------
    path : str
        The full file path to the Excel file (.xlsx, .xls, etc.).
    date_cols : list of str, optional
        A list of column names that should be parsed as datetime objects.
        Defaults to None.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the data from the Excel file.

    Raises
    ------
    FileNotFoundError
        If the file path is invalid or the file does not exist.
    ImportError
        If the required Excel engine (openpyxl or xlrd) is missing.
    RuntimeError
        For general I/O, parsing, or file format errors.
    """

    try:
        if date_cols is not None:
            df = pd.read_excel(path, parse_dates=date_cols, engine='openpyxl')

        else:
            df = pd.read_excel(path, engine='openpyxl')


        if df.empty:
            print(f"Warning: File loaded successfully, but '{path}' is empty")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f'Error: File not found at path: {path}.')

    except Exception as e:
        if 'No such file or directory' in str(e):
            raise FileNotFoundError(f'Error: File not found at path: {path}')

        if 'No engine' in str(e):
            raise ImportError("Error. Missing Excel engine. Please install 'openpyxl' or 'xlrd'.")

        raise RuntimeError(f"A critical error occurred while reading or parsing the Excel file '{path}'."
                          f'Details: {e}')
    


def df_head(df: pd.DataFrame, n_rows: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes and displays a summary of a pandas DataFrame's structure and contents.
    This function provides a quick overview by displaying the first N rows,
    the column data types and non-null counts, and descriptive statistics
    for purely numerical columns (excluding datetime types).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be analyzed and displayed.
    n_rows : int, optional
        The number of rows to display in the header (df.head()). Defaults to 5.

    Returns
    -------
    tuple
        A tuple containing two pandas DataFrames:
        1. df.head(n_rows): The first N rows of the DataFrame.
        2. df.describe(): Descriptive statistics of the DataFrame, calculated only for numeric columns. 

    Raises
    ------
    TypeError
        If the object passed to 'df' is not a pandas DataFrame.
    RuntimeError
        For unexpected internal pandas errors during processing (e.g., in df.describe()).
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'TypeError: Input must be a pandas DataFrame, but received {type(df).__name__}.')

    if not isinstance(n_rows, int) or n_rows < 0:
        print('Warning: n_rows must be a non-negative integer. Defaulting to 5.')
        n_rows = 5

    separator = '\n' + '-'*60 + '\n'

    print(f'--- First {n_rows} Rows of DataFrame ---\n')
    try:
        df_head = safe_display(df.head(n_rows))

    except Exception as e:
        raise RuntimeError(f'Error occurred while processing df.head({n_rows}).Details: {e}.')

    print(separator)

    print('--- Info of DataFrame ---\n')
    try:
        df.info()

    except Exception as e:
        raise RuntimeError(f'Error occurred while processing df.info(). Deatils: {e}.')

    print(separator)

    print('--- Summary Description of DataFrame ---\n')
    try:
        numeric_df = df.select_dtypes(include=np.number, exclude=np.datetime64)
        df_describe = safe_display(numeric_df.describe())

    except Exception as e:
        raise RuntimeError(f'Error occurred while processing df.describe(). Deatils: {e}.')

    print(separator)

    return df_head, df_describe


def col_replace(df: pd.DataFrame, col: str, old_var: Any, new_var: Any) -> pd.DataFrame:
    """
    Replaces old values with new values in a specified DataFrame column,
    returning a new DataFrame.

    This function attempts to replace occurrences of 'old_var' (or a list of
    values) with 'new_var' within the column 'col' of the given DataFrame.
    It ensures the input is valid and raises specific errors on failure.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame in which to perform the replacement.
    col : str
        The name of the column to modify.
    old_var : Any or list of Any
        The value or list of values to be replaced.
    new_var : Any
        The new value to replace 'old_var' with.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified column modified.

    Raises
    ------
    TypeError
        If the first argument is not a pandas DataFrame.
    KeyError
        If the specified column 'col' is not found in the DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if col not in df.columns:
        raise KeyError(f"KeyError: Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}.")

    try:
        new_df = df.copy()

        new_df[col] = new_df[col].replace(old_var, new_var)

        if isinstance(old_var, list) and all(v not in new_df[col].values for v in old_var):
            print(f"Warning: No occurrences of {old_var} found in column '{col}'.")

        elif old_var not in new_df[col].values and not isinstance(old_var, list):
            print(f"Warning: No occurrence of {old_var} found in column '{col}'.")

        return new_df

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during replacement in column '{col}'. Details: {e}.")
    

def null_rows(df: pd.DataFrame, *col: str) -> Union[pd.DataFrame, pd.Series]:
    """
    Identifies and returns boolean indicators for null (NA) values in a DataFrame.

    Checks for null values across the entire DataFrame or specific columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check for null values.
    *cols : str, optional
        Variable number of column names to check. If none are provided,
        all columns are checked.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        A boolean object indicating null values:
        - DataFrame: If zero or multiple columns are specified.
        - Series: If exactly one column is specified.

    Raises
    ------
    TypeError
        If the first argument is not a pandas DataFrame.
    KeyError
        If one or more specified columns are not found in the DataFrame.
    RuntimeError
        For unexpected internal pandas errors.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    col_names = list(col)

    try:
        if not col_names:
            return df.isna()

        elif len(col_names) == 1:
            return df[col_names[0]].isna()

        else:
            return df[col_names].isna()

    except KeyError as e:
        missing_col = e.args[0]
        raise KeyError(f"KeyError: Column(s) not found. Missing column: '{missing_col}'. Available columns: {list(df.columns)}.")

    except Exception as e:
        raise RuntimeError(f'RuntimeError: An unexpected error occurred while processing mull values. Details: {e}.')
    

def df_loc(df: pd.DataFrame, condition: Union[pd.Series, List[bool], Any], col: Union[str, List[str]]) -> Union[pd.DataFrame, pd.Series]:
    """
    Selects rows and columns from a DataFrame based on a boolean condition using `.loc`.

    This function safely applies pandas' `.loc` indexer to filter a DataFrame. It validates
    input types and column existence prior to the operation and provides specific
    error handling for selection issues (KeyError, IndexError, ValueError).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame on which the selection is performed.
    condition : pandas.Series or array-like
        A single-dimensional boolean Series or array used to select rows.
        It must be compatible (e.g., same length/index) with the DataFrame.
    col : str or list of str
        A column name or a list of column names to select.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        A new DataFrame or Series containing the selected data. A Series is returned
        if `col` is a single column name.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame or if `col` is not a string or list of strings.
    KeyError
        If any of the specified columns in `col` are not found in the DataFrame.
    IndexError
        If the length of the boolean `condition` does not match the length of the DataFrame `df`.
    ValueError
        If the `condition` is not a valid boolean array or its dimensions/alignment are incorrect.
    RuntimeError
        For any other unexpected issues during the operation.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if isinstance(col, str):
        cols_to_check = [col]

    elif isinstance(col, list):
        if not all(isinstance(c, str) for c in col):
            raise TypeError(f"TypeError: 'col' must be a string or a list of strings, but received {type(col).__name__}.")

        cols_to_check = col

    else:
        raise TypeError(f"TypeError: 'col' must be a string or a list of strings, but received {type(col).__name__}.")

    missing_cols = [c for c in cols_to_check if c not in df.columns]

    if missing_cols:
        raise KeyError(f'KeyError: The specified column(s) were not found in the DataFrame: {missing_cols}. Available columns: {list(df.columns)}.')

    if not isinstance(condition, (pd.Series, list, np.ndarray)):
        print(f"Warning: 'condition' type ({type(condition).__name__}) may lead to runtime errors. Expecting a pandas Series or array-like.")

    try:
        result = df.loc[condition, col]
        return result

    except KeyError as e:
        raise ValueError(f"ValueError: Invalid index/condition provided. Details: {e}.")
    
    except ValueError as e:
        raise ValueError(f'ValueError: Invalid boolean condition provided. Check that the condition is an array of boolean values and is correctly aligned/dimensioned. Details: {e}.')

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during the `.loc` selection. Details: {e}.")
    

def df_aggfunc(df: pd.DataFrame, aggfunc: Union[str, List[str],
               Callable, Dict[str, Union[str, List[str], Callable]]],
               col: Union[str, List[str], None] = None)-> Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]]:
    """
    Applies an aggregation function or a list of functions to a DataFrame or specified columns.

    This function serves as a flexible wrapper for pandas aggregation methods (.agg)
     and also explicitly handles pandas.Series.value_counts(). It ensures inputs are
    valid and raises specific, informative errors upon failure.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to aggregate.
    aggfunc : str or list of str or callable or dict
        The name of the aggregation function (e.g., 'mean', 'sum'), a list of function
        names, a callable function, or a dictionary mapping columns to functions.
        Also explicitly handles the string 'value_counts'.
    col : str or list of str, optional
        The name of a column or a list of column names on which to perform
        the aggregation. If None, the function is applied to the entire
        DataFrame (except for 'value_counts'). Defaults to None.

    Returns
    -------
    pandas.Series or pandas.DataFrame or dict
        The result of the aggregation. Returns a DataFrame for standard `.agg` or
        a dictionary of Series when applying 'value_counts' to multiple columns.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame or if `col` or `aggfunc` types are invalid.
    ValueError
        If 'value_counts' is specified as `aggfunc` but `col` is not provided,
        or if an aggregation function is incompatible with the data type.
    KeyError
        If any specified column in `col` is not found in the DataFrame.
    RuntimeError
        For any unexpected issues during the aggregation process.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if col is not None:
        cols_to_check: List[str]
        if isinstance(col, str):
            cols_to_check = [col]

        elif isinstance(col, list):
            if not all(isinstance(c, str) for c in col):
                raise TypeError(f"TypeError: List for 'col' must contain only strings. Found non-string element(s).")

            cols_to_check = col

        else:
            raise TypeError(f"TypeError: 'col' must be a string or a list of strings, but received {type(col).__name__}.")

        missing_cols = [c for c in cols_to_check if c not in df.columns]
        if missing_cols:
            raise KeyError(f"KeyError: The specified column(s) were not found in the DataFrame: {missing_cols}. Available columns: {list(df.columns)}.")

    if aggfunc == 'value_counts':
        if col is None:
            raise ValueError(f"ValueError: 'value_counts' requires one or more columns to be specified in 'col'.")

        try:
            if isinstance(col, str):
                return df[col].value_counts()

            else:
                return {c: df[c].value_counts() for c in col}

        except Exception as e:
            raise RuntimeError(f"RuntimeError: Failed to perform 'value_counts' on column(s) {col}. Details: {e}.")

    try:
        df_subset = df[col] if col is not None else df
        return df_subset.agg(aggfunc)

    except KeyError as e:
        raise KeyError(f"KeyError: The specified column(s) were not found in the DataFrame or were referenced incorreclty by aggfunc. Missing key: {e}.")

    except (AttributeError, TypeError) as e:
        raise ValueError(f"ValueError: The aggregation function '{aggfunc}' is not available or compatible with the selected data type(s). Details: {e}.")

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during the aggregation with function '{aggfunc}' on column(s) {col}. Details: {e}.")
    

def drop_labels(df: pd.DataFrame, labels: Union[str, List[str]] = None, axis: Union[int, str, Literal[0, 1, 'index', 'columns']] = 1) -> pd.DataFrame:
    """
    Drops specified labels from rows or columns of a DataFrame, returning a new DataFrame.

    This is a wrapper function for the pandas `drop` method. It performs input validation
    and uses `errors='ignore'` to prevent a KeyError if a specified label does not exist,
    but raises TypeErrors for invalid inputs.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to drop labels.
    labels : str or int or list of str or int, optional  # UPDATED DOCSTRING
        A single label or a list of labels (row or column names) to drop.
        Labels must match the type of the DataFrame's index or columns (usually str or int).
        If None, the function returns a copy of the original DataFrame.
        Defaults to None.
    axis : {0, 1, 'index', 'columns'}, optional
        The axis along which to drop the labels.
        0 or 'index' for rows, 1 or 'columns' for columns.
        Defaults to 1 (columns).

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified labels dropped.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, if `labels` is provided but not a string or list of strings/integers,
        or if `axis` is not a valid type (int or str).
    ValueError
        If `axis` is provided but not one of the valid values (0, 1, 'index', 'columns').
    RuntimeError
        For unexpected issues during the dropping process.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if labels is None:
        return df.copy()

    normalized_labels: List[str]

    if isinstance(labels, str):
        normalized_labels = [labels]

    elif isinstance(labels, list):
        if not all(isinstance(label, (str, int)) for label in labels):
            raise TypeError(f"TypeError: All elements in 'labels' list must be strings or integers.")

        normalized_labels = labels

    else:
        raise TypeError(f"TypeError: 'labels' must be a string or a list of strings, but received {type(labels).__name__}.")

    valid_axis = {0, 1, 'index', 'columns'}

    if axis not in valid_axis:
        if not isinstance(axis, (int, str)):
            raise TypeError(f"TypeError: 'axis' must be an integer (0 or 1) or a string ('index' or 'columns'), but received {type(axis).__name__}.")

        else:
            raise ValueError(f"ValueError: 'axis' must be one of {{0, 1, 'index', 'columns'}}, but received '{axis}'.")

    try:
        return df.drop(labels=normalized_labels, axis=axis, errors='ignore')

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during the drop operation. Details: {e}.")
    


## EDA Functions

def count_plot(title: str, label: str, df: pd.DataFrame, col: str, axis: Literal['x', 'y'] = 'x',
               hue: Optional[str] = None, order: Optional[List[Union[str, float, int]]] = None,
               palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
               tick_rotation: Union[int, float] = 0, ax: Optional[matplotlib.axes._axes.Axes] = None) -> None:
    """
    Creates, labels, and displays a seaborn count plot with bar labels.

    This function is a wrapper around seaborn.countplot, customizing the plot
    with a title, axis labels, value labels (counts), and handling tick rotation.
    Bar labels are centered with white font by default; however, if there are more
    than 10 bars, the labels switch to black font, are positioned at the edge of
    the bars, and use a smaller font size for better visibility.
    It relies on matplotlib (plt) and seaborn (sns).

    Parameters
    ----------
    title : str
        The title of the plot.
    label : str
        The label for the data axis (x-axis if axis='x', y-axis if axis='y').
    df : pandas.DataFrame
        The DataFrame containing the data.
    col : str
        The name of the column to count (plotted on the data axis).
    axis : {'x', 'y'}, optional
        The orientation of the plot ('x' for vertical bars, 'y' for horizontal bars).
        Defaults to 'x'.
    hue : str, optional
        The name of the column for color encoding (grouping). Must be present in `df`.
    order : list of (str, float, or int), optional
        The desired order of the categories on the count axis.
    palette : str, list, or dict, optional
        The color palette to use. Can be a name, list of colors, or a dict mapping hue levels to colors.
    tick_rotation : int or float, optional
        The rotation angle (in degrees) for the tick labels on the data axis. Defaults to 0.
    ax : matplotlib.axes.Axes, optional
        An optional matplotlib Axes object to plot into. If None, a new plot is created internally.

    Returns
    -------
    None
        The function displays the plot using plt.show(), if no external Axes object is provided.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame, or if `title`, `label`, `col`, or `hue` (if provided) are not strings.
    KeyError
        If `col` or `hue` (if provided) is not found in the DataFrame columns.
    ValueError
        If `axis` is not 'x' or 'y', or if `tick_rotation` is not a number.
    RuntimeError
        For unexpected issues during plot generation.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if not all(isinstance(arg, str) for arg in [title, label, col]):
        raise TypeError(f"TypeError: 'title', 'label', and 'col' must all be strings.")

    if col not in df.columns:
        raise KeyError(f"KeyError: Column '{col}' not found in DataFrame. Available columns {list(df.columns)}.")

    if hue is not None:
        if not isinstance(hue, str):
            raise TypeError(f"TypeError: 'hue' must be a string column name.")

        if hue not in df.columns:
            raise KeyError(f"KeyError: Hue column '{hue}' not found in DataFrame. Available columns {list(df.columns)}.")

    if axis not in ('x', 'y'):
        raise ValueError(f"ValueError: 'axis' must be 'x' or 'y', but received '{axis}'.")

    if not isinstance(tick_rotation, (int, float)):
        raise ValueError(f"ValueError: 'tick_rotation' must be a number (int or float), but received {type(tick_rotation).__name__}.")

    try:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,2))
        elif isinstance(ax, np.ndarray):
            if ax.size == 1:
                ax = ax[0]
            else:
                raise ValueError("Expected a single Axes object, but received multiple Axes.")

        plot_df = df.copy()
        plot_df[col] = plot_df[col].astype(str)

        plot_kwargs = {'data': plot_df, 'hue': hue, 'order': order, 'palette': palette}

        if axis == 'x':
            plot_kwargs['x'] = col

        else:
            plot_kwargs['y'] = col

        sns.countplot(ax=ax, **plot_kwargs)

        if axis == 'x':
            ax.set_xlabel(label)
            ax.tick_params(axis='x', rotation=tick_rotation)
            ax.set_ylabel('Count')

        else:
            ax.set_ylabel(label)
            ax.tick_params(axis='y', rotation=tick_rotation)
            ax.set_xlabel('Count')

        for container in ax.containers:
            try:
                n_bars = len(getattr(container, 'patches', []))
            except TypeError:
                n_bars = 0
                
            if n_bars > 10:
                ax.bar_label(container, fmt='%.0f', label_type='edge', color='black', fontsize=6)

            else:
                ax.bar_label(container, fmt='%.0f', label_type='center', color='white', fontsize=10)

        ax.set_title(title)

        if ax is None:
            plt.tight_layout()
            plt.show()

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during plot generation. Details: {e}.")
    

def histogram(title: str, label: str, df: pd.DataFrame, col: str,
              bins: Union[int, List[Union[int, float]]], axis: Literal['x', 'y'] = 'x',
              hue: Optional[str] = None, kde: bool = False, ax: Optional[matplotlib.axes._axes.Axes] = None) -> None:
    """
    Generates and displays a histogram using seaborn.

    This function is a wrapper around seaborn.histplot to visualize the
    distribution of a numerical variable. It provides options for setting the
    number of bins, adding a hue variable for stratification, and overlaying a
    Kernel Density Estimate (KDE) line. The plot is configured with a title
    and a custom label for the data axis. It relies on matplotlib (plt) and seaborn (sns).

    Parameters
    ----------
    title : str
        The title of the histogram.
    label : str
        The label for the **data axis** (x-axis if axis='x', y-axis if axis='y').
    df : pandas.DataFrame
        The DataFrame containing the data.
    col : str
        The name of the numerical column to plot.
    bins : int or list of (int or float)
        The number of bins for the histogram (int) or a sequence of bin edges (list).
    axis : {'x', 'y'}, optional
        The orientation of the plot ('x' for vertical bars, 'y' for horizontal bars).
        Defaults to 'x'.
    hue : str, optional
        The name of a categorical column to use for color encoding (grouping).
        Defaults to None.
    kde : bool, optional
        If True, a Kernel Density Estimate line is overlaid on the plot. Defaults to False.
    ax : matplotlib.axes.Axes, optional
        An optional matplotlib Axes object to plot into. If None, a new plot is created internally.

    Returns
    -------
    None
        The function displays the plot using plt.show(), if no external Axes object is provided.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame, or if `title`, `label`, `col`, or `hue` (if provided) are not strings.
    KeyError
        If `col` or `hue` (if provided) is not found in the DataFrame columns.
    ValueError
        If `axis` is not 'x' or 'y'.
    RuntimeError
        For unexpected issues during plot generation (e.g., non-numerical data).
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if not all(isinstance(arg, str) for arg in [title, label, col]):
        raise TypeError(f"TypeError: 'title', 'label', and 'col' must all be strings.")

    if col not in df.columns:
        raise KeyError(f"KeyError: Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}.")

    if hue is not None:
        if not isinstance(hue, str):
            raise TypeError(f"TypeError: 'hue'must be a string column name.")

        if hue not in df.columns:
            raise KeyError(f"KeyError: Hue column '{hue}' not found in DataFrame. Available columns: {list(df.columns)}.")

    if axis not in ('x', 'y'):
        raise ValueError(f"ValueError: 'axis' must be 'x' or 'y', but received '{axis}'.")

    try:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,2))

        plot_kwargs = {'data': df, 'bins': bins, 'hue': hue, 'kde': kde}

        if axis == 'x':
            plot_kwargs['x'] = col

        else:
            plot_kwargs['y'] = col
        
        sns.histplot(ax=ax, **plot_kwargs)

        if axis == 'x':
            ax.set_xlabel(label)
        else:
            ax.set_ylabel(label)

        ax.set_title(title)

        if ax is None:
            plt.tight_layout()
            plt.show()

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during plot generation. Check if column 'col' is numerical. Details: {e}.")
    

def heatmap(title: str, df: pd.DataFrame, annot: bool = True, cmap: str = 'coolwarm',
            fontsize: Union[int, float] = 7, num_decimals: int = 2, ax: Optional[matplotlib.axes._axes.Axes] = None) -> None:
    """
    Generates and displays a correlation heatmap of a DataFrame.

    This function calculates the correlation matrix of all numeric columns in the
    input DataFrame and visualizes it as a heatmap using seaborn. It provides
    options to show the correlation values on the map, customize the color
    scheme, annotation font size, and decimal precision.

    Parameters
    ----------
    title : str
        The title of the heatmap.
    df : pandas.DataFrame
        The DataFrame for which to create the correlation heatmap.
    annot : bool, optional
        If True, the correlation values are displayed on the heatmap. Defaults to True.
    cmap : str, optional
        The colormap to use for the heatmap. Defaults to 'coolwarm'.
    fontsize : int or float, optional
        The font size of the annotation text (correlation values) on the heatmap.
        Defaults to 7.
    num_decimals : int, optional
        The number of decimal places to display for the correlation values.
        Defaults to 2.
    ax : matplotlib.axes.Axes, optional
        An optional matplotlib Axes object to plot into. If None, a new plot is created internally.

    Returns
    -------
    None
        The function displays the plot using plt.show().

    Raises
    ------
    TypeError
        If `df` is not a DataFrame, or if primary arguments are of the wrong type.
    ValueError
        If `num_decimals` or `fontsize` is negative, or if the DataFrame has no numeric columns.
    RuntimeError
        For unexpected issues during plot generation.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if not isinstance(title, str):
        raise TypeError(f"TypeError: 'title' must be a string, but received {type(title).__name__}.")

    if not isinstance(annot, bool):
        raise TypeError(f"TypeError: 'annot' must be a boolean, but received {type(annot).__name__}.")

    if not isinstance(cmap, str):
        raise TypeError(f"TypeError: 'cmap' must be a string, but received {type(cmap).__name__}.")

    if not isinstance(fontsize, (int, float)) or fontsize <= 0:
        raise ValueError(f"ValueError: 'fontsize' must be a positive number, but received {type(fontsize).__name__}.")

    if not isinstance(num_decimals, int) or num_decimals < 0:
        raise ValueError(f"ValueError: 'num_decimals' must be a non-negative integer, but received {type(num_decimals).__name__}.")

    try:
        corr_matrix = df.corr(numeric_only=True)

        if corr_matrix.empty:
            raise ValueError(f"ValueError: DataFrame contains no numeric columns to calculate a correlation matrix.")
        
        fmt_str = f'.{num_decimals}f'
        created_ax = False

        if ax is None:
            n_cols = corr_matrix.shape[0]
            fig_size = max(6, n_cols / 2)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            created_ax = True

        sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=fmt_str,
                    annot_kws={'fontsize': fontsize}, cbar=True, linewidth=0.5, linecolor='black', ax=ax)

        ax.set_title(title)
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=90)

        if created_ax:
            plt.tight_layout()
            plt.show()

    except ValueError as e:
        raise e

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during plot generation. Details {e}.")
    

def bin_and_plot(title: str, label: str, df: pd.DataFrame, col: str, new_col: str,
                 bins: Union[int, List[Union[int, float]], pd.IntervalIndex],
                 labels: Optional[List[str]] = None, right: bool = True, include_lowest: bool = True,
                 axis: Literal['x', 'y'] = 'x', hue: Optional[str] = None,
                 order: Optional[List[Union[str, float, int]]] = None,
                 palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
                 tick_rotation: Union[int, float] = 0, show_plot: bool = True, ax: Optional[matplotlib.axes._axes.Axes] = None) -> pd.DataFrame:
    """
    Bin a numerical column and optionally generate a count plot of the binned data.

    Uses `pandas.cut` to discretize a numeric column into specified bins and adds
    a new binned column to the DataFrame. Optionally, generates a count plot using
    the `count_plot` function.

    Parameters
    ----------
    title : str
        Title of the count plot.
    label : str
        Label for the data axis (x-axis if axis='x', y-axis if axis='y').
    df : pandas.DataFrame
        The DataFrame containing the column to bin.
    col : str
        Name of the numerical column to bin.
    new_col : str
        Name for the new binned column.
    bins : int, list of numbers, or pandas.IntervalIndex
        Bin edges or number of bins for discretization.
    labels : list of str, optional
        Labels for the returned bins. If provided, also used as the order for plotting.
    right : bool, optional
        Whether bins include the rightmost edge. Default True.
    include_lowest : bool, optional
        Whether the first interval should include the lowest value. Default True.
    axis : {'x', 'y'}, optional
        Orientation of plot bars ('x' for vertical, 'y' for horizontal). Default 'x'.
    hue : str, optional
        Column name for color grouping in the plot.
    order : list, optional
        Desired order of categories on the count axis. Ignored if `labels` is provided.
    palette : str, list, or dict, optional
        Color palette for the plot.
    tick_rotation : int or float, optional
        Rotation of axis tick labels. Default 0.
    show_plot : bool, optional
        Whether to generate and display the plot. Default True.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure/axes is created.

    Returns
    -------
    pd.DataFrame or matplotlib.axes.Axes
        - If `show_plot` is False, returns a copy of the DataFrame with the binned column added.
        - If `show_plot` is True and plotting succeeds, returns the Matplotlib Axes object.
        - If plotting fails or the `count_plot` function is unavailable, returns the DataFrame with the binned column added.

    Raises
    ------
    TypeError
        If input arguments are of incorrect type.
    KeyError
        If `col` or `hue` (if provided) is not found in the DataFrame.
    ValueError
        If binning configuration is invalid (e.g., non-numeric column, mismatched bins/labels).
    RuntimeError
        If unexpected issues occur during binning or plotting.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if not all(isinstance(arg, str) for arg in [title, label, col, new_col]):
        raise TypeError(f"TypeError: 'title', 'label', 'col', and 'new_col' must be strings.")

    if col not in df.columns:
        raise KeyError(f"KeyError: Column '{col}' not found in DataFrame. Available columns {list(df.columns)}.")

    if hue is not None and hue not in df.columns:
        raise KeyError(f"KeyError: Hue column '{hue}' not found in DataFrame. Available columns {list(df.columns)}.")

    if axis not in ('x', 'y'):
        raise ValueError(f"ValueError: 'axis' must be 'x' or 'y', but received {axis}.")

    if not isinstance(show_plot, bool):
        raise TypeError(f"TypeError: 'show_plot' must be boolean.")

    df_new = df.copy()

    try:
        df_new.loc[:, new_col] = pd.cut(df_new[col], bins=bins, labels=labels, right=right,
                                        include_lowest=include_lowest, duplicates='drop')

        if labels is not None:
          plot_order = labels

        elif isinstance(df_new[new_col].dtype, pd.CategoricalDtype):
          plot_order = df_new[new_col].cat.categories.tolist()

        else:
          plot_order = df_new[new_col].unique().tolist()

    except KeyError as e:
        raise KeyError(f"KeyError: An internal column reference failed during binning. Details: {e}.")

    except ValueError as e:
        raise ValueError(f"ValueError: Invalid binning configuration. Check if column '{col}' is numeric, or if bins/labels are correctly formatted. Details: {e}.")

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during binning of column {col}. Details: {e}.")

    if show_plot:
        try:
            count_plot(title=title, label=label, df=df_new, col=new_col, axis=axis, hue=hue, order=plot_order, palette=palette, tick_rotation=tick_rotation, ax=ax)
            return ax

        except NameError:
            warnings.warn("Warning: The 'count_plot' function is required but not defined in the current scope. Plotting skipped.")

        except Exception as e:
            raise RuntimeError(f"RuntimeError: Plotting failed for binned column '{new_col}'. Details: {e}.")

    return df_new


## Statistical Tests Functions

def chi_squared_test(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Performs a Chi-Squared Test of Independence on two categorical columns.

    This function tests for a statistical association between two categorical
    variables using the `pingouin` library. It first checks the assumption that
    less than 20% of the cells have an expected frequency of less than 5. If the
    assumption is met, it calculates and prints the observed and expected frequencies,
    along with the test statistics. It then interprets the results,
    providing a conclusion based on the p-value and the strength of the
    association using Cramer's V.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    col1 : str
        The name of the first categorical column.
    col2 : str
        The name of the second categorical column.
    alpha : float, optional
        The significance level for the test. Defaults to 0.05.

    Returns
    -------
    tuple: A tuple containing:
            - expected (pandas.DataFrame or None): The expected frequencies under the null hypothesis.
            - observed (pandas.DataFrame or None): The observed frequencies (contingency table).
            - stats (pandas.DataFrame or None): A table of test statistics, including
              Chi-Square, p-value, Cramer's V, and statistical power.
            Returns (None, None, None) if assumptions are not met.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame, or if `col1`, `col2`, or `alpha` are of the wrong type.
    KeyError
        If `col1` or `col2` are not found in the DataFrame columns.
    ValueError
        If `alpha` is not between 0 and 1.
    RuntimeError
        For unexpected issues during test calculation.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df'must be a pandas DataFrame, but received {type(df).__name__}.")

    if not all(isinstance(arg, str) for arg in [col1, col2]):
        raise TypeError(f"TypeError: 'col1' and 'col2' must be strings.")

    if col1 not in df.columns:
        raise KeyError(f"KeyError: Column '{col1}' not found in DataFrame. Available columns: {list(df.columns)}.")

    if col2 not in df.columns:
        raise KeyError(f"KeyError: Column '{col2}' not found in DataFrame. Available columns: {list(df.columns)}.")

    if not isinstance(alpha, float) or not (0 < alpha < 1):
        raise ValueError(f"ValueError: 'alpha' must be a float between 0 and 1.")

    try:
        expected, observed, stats = pingouin.chi2_independence(data=df, x=col1 , y=col2)

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during Chi-Squared test: {e}.")


    total_cells = expected.size
    cells_less_5 = (expected.values < 5).sum()
    percent_cells_less_5 = (cells_less_5 / total_cells) * 100

    if percent_cells_less_5 > 20:
        print(f"Assumption Not Met: {percent_cells_less_5}% of cells have expected frequencies < 5 (required < 20%). Test aborted.")
        return None, None, None

    print('------Expected Frequencies------')
    print(expected)

    print('\n------Observed Frequencies------')
    print(observed)

    print('\n------Test Statistics Summary------')
    print(stats)

    try:
        pearson_stats = stats[stats['test'] == 'pearson']
        pearson_p_val = pearson_stats['pval'].item()
        cramer = pearson_stats['cramer'].item()
        statistical_power = pearson_stats['power'].item()

    except IndexError:
        print("Warning: Could not extract Pearson Chi-Squared statistics for interpretation.")
        return expected, observed, stats

    if pearson_p_val <= alpha:
        result = 'Reject Null Hypothesis (H0)'
        if cramer <= 0.10:
            cramer_result = 'Weak Association'
        elif cramer > 0.10 and cramer <= 0.30:
            cramer_result = 'Moderate Association'
        elif cramer > 0.30 and cramer <= 0.50:
            cramer_result = 'Strong Association'
        else:
            cramer_result = 'Very Strong Association'

        type_ii_error_chance = 1 - statistical_power

        print(f'\n--- Conclusion ({col1} vs. {col2}) ---')
        print(f'The p-val is {pearson_p_val:.4f}, which is less than or equal to the significance level (alpha = {alpha}), so we **{result}**.')
        print(f'This means there is a statistically significant association between {col1} and {col2}.')
        print(f'\nThe Cramer\'s V effect size is {cramer:.4f}, indicating a **{cramer_result}** between the two variables.')
        print(f'\nStatistical Power is {statistical_power:.4f}. The chance of committing a Type II error (failing to detect a real effect) is {type_ii_error_chance * 100.0:.2f}%.')

    else:
        result = 'Fail to Reject the Null Hypothesis (H0)'
        print(f'\n--- Conclusion ({col1} vs. {col2}) ---')
        print(f'The p-value is {pearson_p_val:.4f}, which is higher than the significance level (alpha = {alpha}), so we **{result}**.')
        print(f'This means there is no statistically significance evidence of an association between {col1} and {col2}.')

    return expected, observed, stats


## Data Generation Functions

def generate_data(n_records: int = 10000, seed: int = 123) -> pd.DataFrame:
    """
    Generates a synthetic dataset of customer telecommunications data suitable for
    a churn prediction project. The data is structured with realistic dependencies
    and simulated feature interactions to create a realistic classification task.

    The generation process simulates:
    1. Geographic and identifying information (CustomerID, City, Lat/Long).
    2. Demographic and household features (Gender, Senior Citizen, Partner, Dependents).
    3. Contract and Billing terms.
    4. Service dependencies (e.g., Multiple Lines requires Phone Service; Add-ons require Internet Service).
    5. Tenure and financial charges, where Total Charges are dependent on Monthly Charges and Tenure.

    Parameters
    ----------
    n_records (int, optional): The number of customer records to generate. Defaults to 10000.
    seed (int, optional): The random seed used to ensure reproducibility across
                          numpy and the standard random library. Defaults to 123.

    Returns
    -------
    pandas.DataFrame: A DataFrame containing the synthetic customer data with
                      28 columns in a predefined order, suitable for immediate
                      preprocessing and model training.
    """

    np.random.seed(seed)
    random.seed(seed)
    N_RECORDS = n_records

    data: Dict[str, Union[np.ndarray, List[str]]] = {}

    data['CustomerID'] = [f'{i:04d}-CUSTM' for i in range(1, N_RECORDS + 1)]
    data['Count'] = 1
    data['Country'] = 'United States'
    data['State'] = 'California'

    cities = [f'City_{i}' for i in range(1, 51)]
    data['City'] = np.random.choice(cities, N_RECORDS)
    data['Zip Code'] = np.random.randint(90001, 96163, N_RECORDS).astype(str)
    data['Latitude'] = np.random.uniform(32.5, 42.0, N_RECORDS)
    data['Longitude'] = np.random.uniform(-124.5, -114.0, N_RECORDS)
    data['Lat Long'] = [f"{lat:.4f}, {lon:.4f}" for lat, lon in zip(data['Latitude'], data['Longitude'])]


    data['Gender'] = np.random.choice(['Male', 'Female'], N_RECORDS)
    data['Senior Citizen'] = np.random.choice(['Yes', 'No'], N_RECORDS, p=[0.16, 0.84])
    data['Partner'] = np.random.choice(['Yes', 'No'], N_RECORDS)
    data['Dependents'] = np.random.choice(['Yes', 'No'], N_RECORDS, p=[0.3, 0.7])


    data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], N_RECORDS, p=[0.55, 0.22, 0.23])
    data['Paperless Billing'] = np.random.choice(['Yes', 'No'], N_RECORDS)
    data['Payment Method'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], N_RECORDS)

    data['Phone Service'] = np.random.choice(['Yes', 'No'], N_RECORDS, p=[0.90, 0.10])
    data['Internet Service'] = np.random.choice(['DSL', 'Fiber optic', 'No'], N_RECORDS, p=[0.35, 0.45, 0.20])

    data['Multiple Lines'] = np.where(
        data['Phone Service'] == 'No',
        'No phone service',
        np.random.choice(['Yes', 'No'], N_RECORDS, p=[0.45, 0.55])
    )

    internet_dependent_cols = ['Online Security', 'Online Backup', 'Device Protection',
                               'Tech Support', 'Streaming TV', 'Streaming Movies']

    for col in internet_dependent_cols:
        data[col] = np.where(
            data['Internet Service'] == 'No',
            'No internet service',
            np.random.choice(['Yes', 'No'], N_RECORDS, p=[0.4, 0.6])
        )

    data['Tenure Months'] = np.random.randint(1, 73, N_RECORDS)

    monthly_charges = np.where(
        data['Internet Service'] == 'Fiber optic',
        np.random.normal(loc=95, scale=18, size=N_RECORDS).clip(min=50),
        np.random.normal(loc=50, scale=12, size=N_RECORDS).clip(min=18)
    )
    data['Monthly Charges'] = np.round(monthly_charges, 2)

    total_charges = data['Monthly Charges'] * data['Tenure Months'] * np.random.uniform(0.9, 1.1, N_RECORDS)
    data['Total Charges'] = np.round(total_charges, 2)
    data['Total Charges'] = np.where(data['Tenure Months'] <= 1, data['Monthly Charges'], data['Total Charges'])

    churn_prob_base = np.where(data['Contract'] == 'Month-to-month', 0.45, 0.10)
    churn_prob_adjusted = np.where(data['Tenure Months'] <= 5, churn_prob_base * 1.5, churn_prob_base)

    data = pd.DataFrame(data)

    final_order = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long',
        'Latitude', 'Longitude', 'Gender', 'Senior Citizen', 'Partner',
        'Dependents', 'Tenure Months', 'Phone Service', 'Multiple Lines',
        'Internet Service', 'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract',
        'Paperless Billing', 'Payment Method', 'Monthly Charges', 'Total Charges'
    ]

    data = data[final_order]

    return data


def features_to_df(City: Optional[str] = None, Gender: Optional[str] = None, Senior_Citizen: Optional[str] = None, Partner: Optional[str] = None, Dependents: Optional[str] = None,
                   Tenure_Months: Optional[int] = None, Phone_Service: Optional[str] = None, Multiple_Lines: Optional[str] = None, Internet_Service: Optional[str] = None,
                   Online_Security: Optional[str] = None, Online_Backup: Optional[str] = None, Device_Protection: Optional[str] = None, Tech_Support: Optional[str] = None,
                   Streaming_TV: Optional[str] = None, Streaming_Movies: Optional[str] = None, Contract: Optional[str] = None, Paperless_Billing: Optional[str] = None,
                   Payment_Method: Optional[str] = None, Monthly_Charges: Optional[float] = None, Total_Charges: Optional[float] = None):
    """
    Constructs a single-row DataFrame using input customer features, filling in missing values
    based on statistical defaults (mode or median) from the Telco Customer Churn dataset.

    Parameters
    ----------
    All inputs are optional and correspond to customer features:

    City : str, optional
    Gender : str, optional
    Senior_Citizen : str, optional
    Partner : str, optional
    Dependents : str, optional
    Tenure_Months : int, optional
    Phone_Service : str, optional
    Multiple_Lines : str, optional
    Internet_Service : str, optional
    Online_Security : str, optional
    Online_Backup : str, optional
    Device_Protection : str, optional
    Tech_Support : str, optional
    Streaming_TV : str, optional
    Streaming_Movies : str, optional
    Contract : str, optional
    Paperless_Billing : str, optional
    Payment_Method : str, optional
    Monthly_Charges : float, optional
    Total_Charges : float, optional

    Returns
    -------
    pandas.DataFrame
        A single-row DataFrame containing customer features, with missing values filled.

    Raises
    ------
    FileNotFoundError
        If the Telco dataset file cannot be found or loaded.
    RuntimeError
        If there's an unexpected error during feature inference or DataFrame construction.
    """
    
    path = kaggle_download()
    df = read_excel(path)

    try:
        def get_mode(column, condition=None):
            if condition is not None:
                subset = df[condition]
                mode_val = subset[column].mode() if not subset.empty else df[column].mode()
            else:
                mode_val = df[column].mode()
            return mode_val.iloc[0] if not mode_val.empty else None

        # Fills
        City = City or get_mode('City')
        Gender = Gender or get_mode('Gender')
        Senior_Citizen = Senior_Citizen or get_mode('Senior Citizen')
        Partner = Partner or get_mode('Partner')
        Dependents = Dependents or get_mode('Dependents')
        Tenure_Months = Tenure_Months or int(df['Tenure Months'].median())
        Phone_Service = Phone_Service or get_mode('Phone Service')
        Multiple_Lines = Multiple_Lines or get_mode('Multiple Lines', df['Phone Service'] == Phone_Service)
        Internet_Service = Internet_Service or get_mode('Internet Service')

        # Dependent on Internet_Service
        Online_Security = Online_Security or get_mode('Online Security', df['Internet Service'] == Internet_Service)
        Online_Backup = Online_Backup or get_mode('Online Backup', df['Internet Service'] == Internet_Service)
        Device_Protection = Device_Protection or get_mode('Device Protection', df['Internet Service'] == Internet_Service)
        Tech_Support = Tech_Support or get_mode('Tech Support', df['Internet Service'] == Internet_Service)
        Streaming_TV = Streaming_TV or get_mode('Streaming TV', df['Internet Service'] == Internet_Service)
        Streaming_Movies = Streaming_Movies or get_mode('Streaming Movies', df['Internet Service'] == Internet_Service)

        Contract = Contract or get_mode('Contract')
        Paperless_Billing = Paperless_Billing or get_mode('Paperless Billing')
        Payment_Method = Payment_Method or get_mode('Payment Method')
        Monthly_Charges = Monthly_Charges or round(df[df['Internet Service'] == Internet_Service]['Monthly Charges'].median(), 2)
        Total_Charges = Total_Charges or round(Monthly_Charges * Tenure_Months, 2)

        return pd.DataFrame([{
            'City': City,
            'Gender': Gender,
            'Senior Citizen': Senior_Citizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'Tenure Months': Tenure_Months,
            'Phone Service': Phone_Service,
            'Multiple Lines': Multiple_Lines,
            'Internet Service': Internet_Service,
            'Online Security': Online_Security,
            'Online Backup': Online_Backup,
            'Device Protection': Device_Protection,
            'Tech Support': Tech_Support,
            'Streaming TV': Streaming_TV,
            'Streaming Movies': Streaming_Movies,
            'Contract': Contract,
            'Paperless Billing': Paperless_Billing,
            'Payment Method': Payment_Method,
            'Monthly Charges': Monthly_Charges,
            'Total Charges': Total_Charges
        }])

    except Exception as e:
        raise RuntimeError("An error occurred while constructing the customer feature DataFrame.") from e