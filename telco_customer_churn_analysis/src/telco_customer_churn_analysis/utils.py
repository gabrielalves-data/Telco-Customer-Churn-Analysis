import pandas as pd
from typing import Union, Dict, List, Any, Optional, Tuple, Callable, Literal
from IPython import display

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
    

