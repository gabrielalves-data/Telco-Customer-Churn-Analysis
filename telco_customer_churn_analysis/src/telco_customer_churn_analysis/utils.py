import pandas as pd
from typing import Union, Dict, List, Any, Optional, Tuple, Callable, Literal
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Union, Dict, List, Any, Optional, Tuple, Callable, Literal

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
        print(f'Error: Failed to display data. Falling back to print(). Details: {e}.')
        print(data)

    return data


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
            df = pd.read_excel(path, parse_dates=date_cols)

        else:
            df = pd.read_excel(path)


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
    for numerical columns.

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
        2. df.describe(): Descriptive statistics of the DataFrame.

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
        df_describe = safe_display(df.describe())

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
            print(f"Warning: No occurrence of {old_var} found in column '{col}.'")

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
        raise KeyError(f"KeyError: Column(s) not found. Missing column: '{missing_col}'."
                      f"Available columns: {list(df.columns)}.")

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

    try:
        if not isinstance(condition, (pd.Series, list, pd.np.ndarray)):
            print(f"Warning: 'condition' type ({type(condition).__name__}) may lead to runtime errors. Expecting a pandas Series or array-like.")

    except AttributeError:
        pass

    try:
        result = df.loc[condition, col]
        return result

    except IndexError as e:
        length = f"Condition length: {len(condition) if hasattr(condition, '__len__') else 'N/A'}, DataFrame length: {len(df)}."
        raise IndexError(f'IndexError: The length of the boolean condition does not match the DataFrame length. {length} Details: {e}.')

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

    except AttributeError as e:
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
    labels : str or list of str, optional
        A single label or a list of labels (row or column names) to drop.
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
        If `df` is not a pandas DataFrame, if `labels` is provided but not a string or list of strings,
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
        if not all(isinstance(label, str) for label in labels):
            raise TypeError(f"TypeError: All elements in 'labels' list must be strings.")

        normalized_labels = labels

    else:
        raise TypeError(f"TypeError: 'labels' must be a string or a list of strings, but received {type(labels).__name__}.")

    valid_axis = {0, 1, 'index', 'columns'}

    if axis not in valid_axis:
        if not isinstance(axis, (int, str)):
            raise TypeError(f"TypeError: 'axis' must be an integer (0 or 1) or a string ('index' or 'columns'), but received {type(axis).__name__}.")

        else:
            raise ValueError(f"ValueError: 'axis' must be one of {valid_axis}, but received '{axis}'.")

    try:
        return df.drop(labels=normalized_labels, axis=axis, errors='ignore')

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during the drop operation. Details: {e}.")
    


## EDA Functions

def count_plot(title: str, label: str, df: pd.DataFrame, col: str, axis: Literal['x', 'y'] = 'x',
               hue: Optional[str] = None, order: Optional[List[Union[str, float, int]]] = None,
               palette: Optional[Union[str, List[str], Dict[str, str]]] = None,
               tick_rotation: Union[int, float] = 0) -> None:
    """
    Creates, labels, and displays a seaborn count plot with bar labels.

    This function is a wrapper around seaborn.countplot, customizing the plot
    with a title, axis labels, value labels (counts) centered in the bars,
    and handling tick rotation. It relies on matplotlib (plt) and seaborn (sns).

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

    Returns
    -------
    None
        The function displays the plot using plt.show().

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
        fig, ax = plt.subplots(figsize=(10,6))

        plot_kwargs = {'data': df, 'hue': hue, 'order': order, 'palette': palette, axis: col}

        sns.countplot(ax=ax, **plot_kwargs)

        if axis == 'x':
            ax.set_xlabel(label)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=tick_rotation)
            ax.set_ylabel('Count')

        else:
            ax.set_ylabel(label)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=tick_rotation)
            ax.set_xlabel('Count')

        for container in ax.containers:
          ax.bar_label(container, fmt='%.0f', label_type='center', color='white', fontsize=10)

        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during plot generation. Details: {e}.")
    

def histogram(title: str, label: str, df: pd.DataFrame, col: str,
              bins: Union[int, List[Union[int, float]]], axis: Literal['x', 'y'] = 'x',
              hue: Optional[str] = None, kde: bool = False) -> None:
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

    Returns
    -------
    None
        The function displays the plot using plt.show().

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
        fig, ax = plt.subplots(figsize=(10,6))

        plot_kwargs = {'data': df, 'bins': bins, 'hue': hue, 'kde': kde, axis: col}

        sns.histplot(ax=ax, **plot_kwargs)

        if axis == 'x':
            plt.xlabel(label)
        else:
            plt.ylabel(label)

        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during plot generation. Check if column 'col' is numerical. Details: {e}.")
    

def heatmap(title: str, df: pd.DataFrame, annot: bool = True, cmap: str = 'coolwarm',
            fontsize: Union[int, float] = 7, num_decimals: int = 2) -> None:
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

        n_cols = corr_matrix.shape[0]
        fig_size = max(6, n_cols / 2)
        plt.figure(figsize=(fig_size, fig_size))

        fmt_str = f'.{num_decimals}f'

        sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=fmt_str,
                    annot_kws={'fontsize': fontsize}, cbar=True, linewidth=0.5, linecolor='black')

        plt.title(title)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

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
                 tick_rotation: Union[int, float] = 0, show_plot: bool = True) -> pd.DataFrame:
    """
    Bins a numerical column and optionally generates a count plot of the binned data.

    This function first uses pandas.cut to discretize a numerical column into
    specified bins and returns a new DataFrame with the binned column added.
    It then optionally calls the `count_plot` function to visualize the distribution.

    Parameters
    ----------
    title : str
        The title of the count plot.
    label : str
        The label for the data axis (x-axis if axis='x', y-axis if axis='y').
    df : pandas.DataFrame
        The DataFrame containing the data.
    col : str
        The name of the numerical column to be binned.
    new_col : str
        The name for the new binned column.
    bins : int, list of (int/float), or pandas.IntervalIndex
        The criteria for binning.
    labels : list of str, optional
        The labels for the returned bins. If provided, they will also be used
        as the order for the count plot, overriding `order`.
    right : bool, optional
        Indicates whether the bins include the rightmost edge. Defaults to True.
    include_lowest : bool, optional
        Indicates whether the first bin should include the lower bound. Defaults to True.
    axis : {'x', 'y'}, optional
        The orientation of the plot ('x' for vertical bars, 'y' for horizontal bars).
        Defaults to 'x'.
    hue : str, optional
        The name of the column for color encoding (grouping). Defaults to None.
    order : list of (str, float, or int), optional
        **Deprecated/Overridden.** The desired order of the categories (binned labels)
        on the count axis. Note: If `labels` is provided, it will be used as the order.
    palette : str, list, or dict, optional
        The color palette to use for the count plot. Defaults to None.
    tick_rotation : int or float, optional
        The rotation angle for axis ticks. Defaults to 0.
    show_plot : bool, optional
        If True, the count plot is generated and displayed. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame with the new binned column (`new_col`) added.

    Raises
    ------
    TypeError
        If input arguments (like `df`, `title`, `col`, `new_col`, `hue`, `bins`, `labels`)
        are not of the correct type.
    KeyError
        If `col` or `hue` (if provided) is not found in the DataFrame columns.
    ValueError
        If binning configuration is invalid (e.g., non-numeric column, mismatched bins/labels).
    RuntimeError
        For unexpected issues during binning or plotting.
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
                                        include_lowest=include_lowest)

        if labels is not None:
          plot_order = labels

        elif pd.api.types.is_categorical_dtype(df_new[new_col]):
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
        count_plot(title=title, label=label, df=df_new, col=new_col, axis=axis, hue=hue, order=plot_order, palette=palette, tick_rotation=tick_rotation)

      except NameError:
        warnings.warn("Warning: The 'count_plot' function is required but not defined in the current scope. Plotting skipped.")

      except Exception as e:
        raise RuntimeError(f"RuntimeError: Plotting failed for binned column '{new_col}'. Details: {e}.")

    return df_new


