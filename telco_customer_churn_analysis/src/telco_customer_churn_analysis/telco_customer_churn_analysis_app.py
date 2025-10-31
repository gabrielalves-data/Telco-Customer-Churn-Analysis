import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from typing import Optional
import os
from flask import current_app
import io
from contextlib import redirect_stdout
import joblib
import traceback

from .model_utils import (preprocess_data, model_pipelines, hyperparameters, train_evaluate_model,
                                                           voting_model, comparative_models, profit_curve, deployment_model,
                                                           predict_churn, abc_test
                                                           )

from .utils import (kaggle_download, safe_display, read_excel, df_head,
                                                     col_replace, null_rows, df_loc, df_aggfunc,
                                                     drop_labels, count_plot, histogram, heatmap,
                                                     bin_and_plot, chi_squared_test, generate_data, features_to_df)

from .model_xai import (model_global_explainer, model_local_explainer)

from .telco_customer_churn_analysis import (data_preprocessing, generate_test_data, bin_df, get_model, deploy_model,
                                            features_to_dataframe, predict_df, abc_test_assignment, profit_curve_threshold,
                                            global_explainer, local_explainer)


def data_preprocessing_app():
    """
    Download, clean, and preprocess the Telco Customer Churn dataset.

    This function checks if the dataset exists locally. If not, it downloads
    the dataset via Kaggle. It then loads the data into a DataFrame, cleans
    the 'Total Charges' column, fills missing values, reduces the cardinality
    of the 'City' column, and drops irrelevant columns. Summary statistics
    are printed to the console for basic inspection.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        - The preprocessed DataFrame ready for further analysis.
        - A string containing printed summary statistics captured during processing.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found or downloaded.
    ValueError
        If conversion to numeric fails for 'Total Charges'.
    """

    full_path = kaggle_download()
    df = read_excel(full_path)

    df['Total Charges'] = df['Total Charges'].astype(str).str.strip()
    df = col_replace(df, 'Total Charges', 'nan', np.nan)
    df = col_replace(df, 'Total Charges', '', np.nan)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    total_charges_null = null_rows(df, 'Total Charges')
    total_charges = df_loc(df, total_charges_null, 'Monthly Charges') * df_loc(df, total_charges_null, 'Tenure Months')
    df.loc[total_charges_null, 'Total Charges'] = total_charges

    top_20_cities = df['City'].value_counts().nlargest(20).index
    df['City'] = df['City'].apply(lambda x: x if x in top_20_cities else 'Other')

    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        df_head(df)
        
        missing_rows = null_rows(df).sum()
        print('--- Number of Missing Values by Column ---')
        print(missing_rows)

        churn_counts = df_aggfunc(df, 'value_counts', 'Churn Value')
        print('--- Number of Rows by Churn Value ---')
        print(churn_counts)

        mean_churn = df_aggfunc(df, 'mean', 'Churn Value')
        print('--- Mean of Churn Value ---')
        print(mean_churn)

        median_monthly = df_aggfunc(df, 'median', 'Monthly Charges')
        print('--- Median of Monthly Charges ---')
        print(median_monthly)

    df = drop_labels(df, ['CustomerID', 'Count', 'Country', 'Lat Long', 'Churn Label'])

    printed_text = output_buffer.getvalue()

    return df, printed_text


def exploratory_analysis_app(df: pd.DataFrame):
    """
    Perform exploratory data analysis (EDA) and generate distribution plots for Telco customer data.

    This function creates count plots and histograms for categorical and numerical features, 
    including churn-related variables. It also generates binned plots for tenure, churn probability, 
    and customer value. Plots are saved in pages under the Flask app's `static/eda` folder for visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed Telco customer dataset.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        - The DataFrame potentially augmented with additional binned columns.
        - A list of filenames corresponding to the saved EDA plots.

    Raises
    ------
    KeyError
        If expected columns are missing from the DataFrame.
    RuntimeError
        If plotting or binning encounters an unexpected error.
    """

    static_eda_dir = os.path.join(current_app.root_path, 'static', 'eda')
    os.makedirs(static_eda_dir, exist_ok=True)

    plot_funcs = []

    top_20_cities = df['City'].value_counts().head(20).index
    filtered_df = df[df['City'].isin(top_20_cities)]

    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by City', 'City', filtered_df, 'City', order=top_20_cities, tick_rotation=45 ,ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Gender', 'Gender', df, 'Gender', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Seniority', 'Senior Citizen', df, 'Senior Citizen', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having a Partner', 'Partner', df, 'Partner', ax=ax))
    plot_funcs.append(lambda ax: histogram('Distribution of Customers by Tenure Months', 'Tenure Months', df, 'Tenure Months', 20, ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Phone Service', 'Phone Service', df, 'Phone Service', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Multiple Lines', 'Multiple Lines', df, 'Multiple Lines', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Internet Service', 'Internet Service', df, 'Internet Service', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Online Security Service', 'Online Security', df, 'Online Security', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Online Backup Service', 'Online Backup', df, 'Online Backup', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Device Protection Service', 'Device Protection', df, 'Device Protection', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Tech Support Service', 'Tech Support', df, 'Tech Support', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Dependents', 'Dependents', df, 'Dependents', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Streaming TV Service', 'Streaming TV', df, 'Streaming TV', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having Streaming Movies Service', 'Streaming Movies', df, 'Streaming Movies', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Contract', 'Contract', df, 'Contract', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Using Paperless Billing', 'Paperless Billing', df, 'Paperless Billing', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Payment Method', 'Payment Method', df, 'Payment Method', tick_rotation=90, ax=ax))
    plot_funcs.append(lambda ax: histogram('Distribution of Monthly Charges', 'Monthly Charges', df, 'Monthly Charges', 20, ax=ax))
    plot_funcs.append(lambda ax: histogram('Distribution of Total Charges', 'Total Charges', df, 'Total Charges', 20, ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Churn Value', 'Churn Value', df, 'Churn Value', ax=ax))
    plot_funcs.append(lambda ax: histogram('Distribution of Churn Score', 'Churn Score', df, 'Churn Score', 20, ax=ax))
    plot_funcs.append(lambda ax: histogram('Distribution of CLTV', 'CLTV', df, 'CLTV', 20, ax=ax))

    churn_reason_order = df_aggfunc(df, 'value_counts', 'Churn Reason').index
    plot_funcs.append(lambda ax: count_plot('Distribution of Churned Customers by Churn Reason', 'Churn Reason', df, 'Churn Reason',
               order=churn_reason_order, tick_rotation=90, ax=ax))
    
    plot_funcs.append(lambda ax: heatmap('Numeric Features Heatmap', df, ax=ax))

    df_churn = df[df['Churn Value'] == 1]

    plot_funcs.append(lambda ax: histogram('Distribution of Churned Customers Monthly Charges', 'Monthly Charges', df_churn, 'Monthly Charges', 20, ax=ax))

    plot_funcs.append(lambda ax: histogram('Distribution of Churned Customers Tenure Months', 'Tenure Months', df_churn, 'Tenure Months', 20, ax=ax))

    plot_funcs.append(lambda ax: histogram('Distribution of Churned Customers Churn Score', 'Churn Score', df_churn, 'Churn Score', 20, ax=ax))

    print('\nMedian CLTV of churn users that have >= 70 Tenure Months: ' + str(df_churn[df_churn['Tenure Months'] >= 70]['CLTV'].median()) + '\n')
    plot_funcs.append(lambda ax: histogram('Distribution of Churned Customers CLTV', 'CLTV', df_churn, 'CLTV', 20, ax=ax))

    df = bin_and_plot('Distribution of Customers Grouped by Tenure Group', 'Tenure Group', df, 'Tenure Months', 'Tenure Group',
                      [0, 12, 30, 50, df['Tenure Months'].max()],
                      ['New Customer', 'New/Established Customer', 'Established/Veteran Customer', 'Veteran Customer'], show_plot=False)
    
    plot_funcs.append(lambda ax: bin_and_plot('Distribution of Customers Grouped by Tenure Group', 'Tenure Group', df, 'Tenure Months', 'Tenure Group',
                      [0, 12, 30, 50, df['Tenure Months'].max()],
                      ['New Customer', 'New/Established Customer', 'Established/Veteran Customer', 'Veteran Customer'], tick_rotation=45,
                      ax=ax))
    
    df = bin_and_plot('Distribution of Customers Grouped by Probability of Churn', 'Churn Probability', df, 'Churn Score',
                      'Churn Probability', [0, 25, 50, 75, 100],
                      ['Less Probability of Churn', 'Less/Moderate Probability of Churn', 'Moderate/High Probability of Churn', 'High Probability of Churn'],
                      tick_rotation=10, show_plot=False)
    
    plot_funcs.append(lambda ax: bin_and_plot('Distribution of Customers Grouped by Probability of Churn', 'Churn Probability', df, 'Churn Score',
                      'Churn Probability', [0, 25, 50, 75, 100],
                      ['Less Probability of Churn', 'Less/Moderate Probability of Churn', 'Moderate/High Probability of Churn', 'High Probability of Churn'],
                      tick_rotation=10, ax=ax))
    
    df = bin_and_plot('Distribution of Customers Grouped by Customer Value', 'Customer Value', df, 'CLTV', 'Customer Value',
                      [0, 3000, 4000, 5000, df['CLTV'].max()],
                      ['Low Value', 'Low/Mid Value', 'Mid/High Value', 'High Value'], show_plot=False)
    
    plot_funcs.append(lambda ax: bin_and_plot('Distribution of Customers Grouped by Customer Value', 'Customer Value', df, 'CLTV', 'Customer Value',
                      [0, 3000, 4000, 5000, df['CLTV'].max()],
                      ['Low Value', 'Low/Mid Value', 'Mid/High Value', 'High Value'], ax=ax))
    
    plot_counts = len(plot_funcs)
    plots_per_page = 6
    cols = 3
    total_pages = (plot_counts + plots_per_page - 1) // plots_per_page

    image_filenames = []
    
    for page in range(total_pages):
        start_idx = page * plots_per_page
        end_idx = min(start_idx + plots_per_page, plot_counts)

        current_plots = plot_funcs[start_idx:end_idx]
    
        num_plots = len(current_plots)
        rows = (num_plots + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
        
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        
        else:
            axs = np.array([axs])

        for i, plot_fn in enumerate(current_plots):
            plot_fn(axs[i])
            axs[i].title.set_fontsize(9)
            axs[i].tick_params(axis='x', labelsize = 6)

        for j in range(num_plots, len(axs)):
            axs[j].axis('off')
            
        plt.tight_layout(pad=3.0, w_pad=2.5, h_pad=3.5)

        filename = f"eda_page_{page + 1}.png"
        filepath = os.path.join(static_eda_dir, filename)
        plt.savefig(filepath)
        plt.close()

        image_filenames.append(filename)

    return df, image_filenames


def bin_df_app(df: pd.DataFrame, show: bool = False):
    """
    Bin the 'Tenure Months' column into categorical groups and optionally display a plot.

    This function bins the 'Tenure Months' column into customer tenure groups:
    'New Customer', 'New/Established Customer', 'Established/Veteran Customer', and 'Veteran Customer'.
    If `show=True`, it generates a count plot for the binned column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the 'Tenure Months' column.
    show : bool, optional
        If True, generates and displays a count plot for the binned column. Default is False.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with a new column 'Tenure Group' added.

    Raises
    ------
    KeyError
        If the 'Tenure Months' column is missing from the DataFrame.
    """

    def safe_max(series: pd.Series, default: int):
        """Helper: Return max if numeric and not nan; else default."""
        try:
            val = series.max()
            if pd.isna(val) or val <= 0:
                return default
            return val
        except Exception:
            return default
        
    def create_bins(series: pd.Series, base_bins, labels):
        series_max = safe_max(series, base_bins[-1])
        bins = base_bins.copy()
        if series_max > bins[-1]:
            bins.append(series_max + 1)
        bins = sorted(set(bins))
        lbls = labels[:len(bins) - 1]

        return bins, lbls

    if show:
        plots_count = 3
        cols = 3
        rows = (plots_count + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axs = np.atleast_1d(axs).flatten()

    else:
        axs = [None, None, None]

    df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce')
    tenure_bins, tenure_labels = create_bins(
        df['Tenure Months'],
        base_bins=[0, 12, 30, 50],
        labels = ['New Customer', 'New/Established Customer', 'Established/Veteran Customer', 'Veteran Customer']
    )

    df = bin_and_plot('Distribution of Customers Grouped by Tenure Group', 'Tenure Group', df, 'Tenure Months', 'Tenure Group',
                      tenure_bins,
                      tenure_labels,
                      show_plot=show, ax=axs[0] if show else None)
    
    if show is True:
        for ax in axs:
            ax.title.set_fontsize(9)
            ax.tick_params(axis='x', labelsize=6)
        
        plt.tight_layout()
        plt.show()

    return df



def hypothesis_test_app(data_choice: str = 'Test', col1: str = None, col2: str = None):
    """
    Perform a chi-squared test of independence between two columns.

    Depending on `data_choice`, this function uses either preprocessed test data
    or newly generated synthetic data. It bins the data before running the chi-squared test
    and captures the output.

    Parameters
    ----------
    data_choice : str, default='Test'
        - 'Test': Use preprocessed test dataset.
        - 'New': Use new synthetic dataset.
    col1 : str, optional
        First column to test. Defaults to first relevant column.
    col2 : str, optional
        Second column to test. Defaults to second relevant column.

    Returns
    -------
    str
        Captured text output of the chi-squared test results.

    Raises
    ------
    KeyError
        If the selected columns are missing from the DataFrame.
    ValueError
        If `data_choice` is invalid or DataFrame has insufficient columns.
    """

    if data_choice == 'Test':
        print('Use test data')
        df = data_preprocessing()
        df = bin_df(df)

    elif data_choice == 'New':
        print('Use new data')
        df = generate_test_data()
        df = bin_df_app(df)

    else:
        raise ValueError('Input invalid. Please select "Test" to use the training data or "New" to use new generated data.')

    try:
        if col1 is None or col1 not in df.columns:
            if len(df.columns) >= 1:
                col1 = df.columns[0]
            else:
                raise ValueError("DataFrame has no columns.")
        
        if col2 is None or col2 not in df.columns:
            if len(df.columns) >= 2:
                col2 = df.columns[1]
            else:
                raise ValueError('DataFrame has less than 2 columns.')
    
    except ValueError as e:
        raise ValueError(f"Please select an dataframe column.")
    
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        _ , _ , _ = chi_squared_test(df, col1, col2)

    return buffer.getvalue()


def train_evaluate_deploy_app():
    """
    Train, evaluate, and deploy a churn classification model.

    This function performs the full pipeline of preprocessing data,
    binning relevant columns, training multiple models, selecting the best model,
    evaluating performance, and deploying the model for predictions.

    Returns
    -------
    str
        Captured printed output including training, evaluation, and deployment logs.

    Raises
    ------
    Exception
        If model training or deployment fails.
    """
    output_buffer = io.StringIO()

    print('Start Training and Evaluate -- App')
    df = data_preprocessing()
    df = bin_df(df)

    with redirect_stdout(output_buffer):
        try:
            all_models, all_results, best_model, X_train, X_test, y_test = get_model(df)
            print('Model Created')

        except Exception as e:
            print(f"Error while loading or training model: {e}")
            traceback.print_exc()
            raise

        deploy_model(
            df, best_model, target='Churn Value',
            cols_to_drop=[
                'State', 'Zip Code', 'Latitude', 'Longitude','Churn Value', 'Churn Score', 'CLTV', 'Churn Reason',
                'Churn Probability', 'Customer Value', 'Tenure Months','Total Charges']
                )
        
    return output_buffer.getvalue()


def predict_with_best_profit_threshold_app(df=None, y_test=None, aggfunc: str = 'median', col: str = 'CLTV', cost: float = 100.0,
                                       retention_rate: float = 0.8, abc_assignment = False,
                                       City: Optional[str] = None, Gender: Optional[str] = None, Senior_Citizen: Optional[str] = None,
                                       Partner: Optional[str] = None, Dependents: Optional[str] = None, Tenure_Months: Optional[int] = None,
                                       Phone_Service: Optional[str] = None, Multiple_Lines: Optional[str] = None,
                                       Internet_Service: Optional[str] = None, Online_Security: Optional[str] = None,
                                       Online_Backup: Optional[str] = None, Device_Protection: Optional[str] = None,
                                       Tech_Support: Optional[str] = None, Streaming_TV: Optional[str] = None,
                                       Streaming_Movies: Optional[str] = None, Contract: Optional[str] = None,
                                       Paperless_Billing: Optional[str] = None, Payment_Method: Optional[str] = None,
                                       Monthly_Charges: Optional[float] = None, Total_Charges: Optional[float] = None):
    """
    Predict churn using the best threshold determined by the profit curve.

    This function optionally accepts user-provided features or a full DataFrame.
    It bins the data, loads the deployed model, calculates the optimal threshold
    based on profit considerations, and predicts churn outcomes.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input dataset. If None, new synthetic data will be generated.
    y_test : pd.Series, optional
        True labels for evaluation. If None, loaded from the deployed model.
    aggfunc : str, default='median'
        Aggregation function used for profit calculation ('mean' or 'median').
    col : str, default='CLTV'
        Column representing customer lifetime value.
    cost : float, default=100.0
        Cost to retain a customer.
    retention_rate : float, default=0.8
        Expected customer retention rate.
    abc_assignment : bool, default=False
        If True, applies ABC assignment to predicted customers.
    City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
    Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
    Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges : optional
        Individual customer feature values for prediction.

    Returns
    -------
    tuple
        threshold : float
            Optimal churn prediction threshold from the profit curve.
        predicted_df_html : str
            HTML table of the predicted churn outcomes with optional ABC assignment.

    Raises
    ------
    FileNotFoundError
        If the deployed model file ('model_results.pkl') cannot be found.
    ValueError
        If feature inputs are invalid or incompatible with the model.
    """

    def clean_features(*args):
        return [val if val not in ("", None) else None for val in args]
    
    features_provided = any(val not in (None, "", 0, False) for val in [
        City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
        Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
        Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges
        ])
    
    print('-- Get features provided')
    
    if features_provided:
        clean_vals = clean_features(City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
                        Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
                        Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges)
        
        df = features_to_dataframe(*clean_vals)

        print('Features cleaned')
        print(df)
    
    elif df is not None:
        df = df.copy()
    
    else:
        df = generate_test_data()

    print('before bin')
    df = bin_df_app(df)
    print('after bin')

    print('--- User Df ---')
    print(df)

    if y_test is None:
        with open('model_results.pkl', 'rb') as deployed_model:
            bundle = joblib.load(deployed_model)
        
        y_test = bundle['y_test']

    with open('model_results.pkl', 'rb') as deployed_model:
        bundle = joblib.load(deployed_model)
    
    model_df = bundle['all_results']
    
    threshold = profit_curve_threshold(aggfunc, col, model_df, cost, retention_rate, y_test)

    print('--- Predict ---')
    predicted_df = predict_df(df, threshold)
    print('Model deployed successfully!')
    print(f'Best threshold from profit curve: {threshold}')
    
    if abc_assignment:
        predicted_df = abc_test_assignment(predicted_df)
        print('Prediction Results with ABC assignment')
        print(predicted_df)

    return threshold, predicted_df.to_html(classes='table table-striped', index=False)


def predict_with_xai_app(df = None, threshold_input: float = 0.5, 
                         global_xai: bool = False,
                         local_xai: bool = False,
                         index_local: int = 0,
                         City: Optional[str] = None, Gender: Optional[str] = None, Senior_Citizen: Optional[str] = None,
                         Partner: Optional[str] = None, Dependents: Optional[str] = None, Tenure_Months: Optional[int] = None,
                         Phone_Service: Optional[str] = None, Multiple_Lines: Optional[str] = None,
                         Internet_Service: Optional[str] = None, Online_Security: Optional[str] = None,
                         Online_Backup: Optional[str] = None, Device_Protection: Optional[str] = None,
                         Tech_Support: Optional[str] = None, Streaming_TV: Optional[str] = None,
                         Streaming_Movies: Optional[str] = None, Contract: Optional[str] = None,
                         Paperless_Billing: Optional[str] = None, Payment_Method: Optional[str] = None,
                         Monthly_Charges: Optional[float] = None, Total_Charges: Optional[float] = None):
    """
    Predict customer churn using a specified threshold and optionally generate XAI explanations.

    The function accepts either a full DataFrame or individual customer feature inputs. 
    It bins the data, applies the deployed model to predict churn, and can generate
    global or local explainable AI (XAI) outputs.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input dataset for prediction. If None, new synthetic data will be generated.
    threshold_input : float, default=0.5
        Threshold for converting predicted probabilities into binary churn predictions.
    global_xai : bool, default=False
        If True, generates global model explanations.
    local_xai : bool, default=False
        If True, generates local explanation for a single data point.
    index_local : int, default=0
        Index of the data point to explain when `local_xai` is True.
    City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
    Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
    Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges : optional
        Individual customer feature values. If provided, a DataFrame will be created from these features 
        for prediction.

    Returns
    -------
    str
        HTML table of predicted churn results.

    Raises
    ------
    FileNotFoundError
        If the deployed model or results file cannot be found.
    ValueError
        If feature inputs are invalid or incompatible with the deployed model.
    """
    
    def clean_features(*args):
        return [val if val not in ("", None) else None for val in args]
    
    features_provided = any(val not in (None, "", 0, False) for val in [
        City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
        Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
        Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges
        ])
    
    print('-- Get features provided')
    
    if features_provided:
        clean_vals = clean_features(City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
                        Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
                        Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges)
        
        df = features_to_dataframe(*clean_vals)

        print('Features cleaned')
        print(df)
    
    elif df is not None:
        df = df.copy()
    
    else:
        df = generate_test_data()

    print('before bin')
    df = bin_df_app(df)
    print('after bin')

    print('--- User Df ---')
    print(df)

    predicted_df = predict_df(df, threshold_input)

    with open('deployment_pipeline.pkl', 'rb') as deployed_model:
        bundle = joblib.load(deployed_model)
        
    model = bundle

    print('No xai request')

    if global_xai:
        print('Got global xai requests')
        with open('model_results.pkl', 'rb') as deployed_model:
            bundle = joblib.load(deployed_model)

        X_train, X_test = bundle['X_train'], bundle['X_test']

        global_explainer(model, X_train, X_test)

    if local_xai:
        print('Got local xai requests')
        with open('model_results.pkl', 'rb') as deployed_model:
            bundle = joblib.load(deployed_model)

        X_train, X_test = bundle['X_train'], bundle['X_test']

        local_explainer(model, X_train, X_test, index_local)


    return predicted_df.to_html(classes="table table-bordered table-striped", index=False)
