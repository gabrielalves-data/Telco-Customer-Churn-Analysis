import pandas as pd
import os
import kagglehub
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from .model_utils import (preprocess_data, model_pipelines, hyperparameters, train_evaluate_model,
                                                           voting_model, comparative_models, profit_curve, deployment_model,
                                                           predict_churn, abc_test
                                                           )
from .utils import (safe_display, read_excel, df_head,
                                                     col_replace, null_rows, df_loc, df_aggfunc,
                                                     drop_labels, count_plot, histogram, heatmap,
                                                     bin_and_plot, chi_squared_test, generate_data)
from .model_xai import (model_global_explainer, model_local_explainer)

def data_preprocessing():
    """
    Download, clean, and preprocess the Telco Customer Churn dataset.

    This function downloads the dataset from Kaggle, loads it into a DataFrame,
    cleans the 'Total Charges' column by handling missing or malformed data,
    calculates missing 'Total Charges' values, drops irrelevant columns, and
    prints summaries about missing values and churn statistics.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame ready for further analysis.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be downloaded or found.
    ValueError
        If data conversion to numeric fails for 'Total Charges'.
    """
    path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")
    filename = 'Telco_customer_churn.xlsx'
    full_path = os.path.join(path, filename)

    df = read_excel(full_path)
    df_head(df)

    df['Total Charges'] = df['Total Charges'].astype(str).str.strip()
    df = col_replace(df, 'Total Charges', 'nan', np.nan)
    df = col_replace(df, 'Total Charges', '', np.nan)
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    total_charges_null = null_rows(df, 'Total Charges')
    total_charges = df_loc(df, total_charges_null, 'Monthly Charges') * df_loc(df, total_charges_null, 'Tenure Months')
    df.loc[total_charges_null, 'Total Charges'] = total_charges

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

    return df


def exploratory_analysis(df):
    """
    Perform exploratory data analysis and generate multiple distribution plots.

    This function creates a series of count plots and histograms for various
    categorical and numerical features, including churn-related variables.
    It also generates binned plots for tenure, churn probability, and customer value.
    The plots are displayed in pages for better visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed Telco customer dataset.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame potentially modified with additional binned columns.

    Raises
    ------
    KeyError
        If expected columns are missing from the DataFrame.
    """
    plot_funcs = []

    top_10_cities = df['City'].value_counts().head(10).index
    filtered_df = df[df['City'].isin(top_10_cities)]

    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by City', 'City', filtered_df, 'City', order=top_10_cities, tick_rotation=45 ,ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Gender', 'Gender', df, 'Gender', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Seniority', 'Senior Citizen', df, 'Senior Citizen', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Having a Partner', 'Partner', df, 'Partner', ax=ax))
    plot_funcs.append(lambda ax: count_plot('Distribution of Customers by Tenure Months', 'Tenure Months', df, 'Tenure Months', tick_rotation=90, ax=ax))
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

    print('--- Distribution of Monthly Charges of Churned Customers ---')
    plot_funcs.append(lambda ax: histogram('Distribution of Churned Customers Monthly Charges', 'Monthly Charges', df_churn, 'Monthly Charges', 20, ax=ax))

    print('--- Distribution of Tenure Months of Churned Customers ---')
    plot_funcs.append(lambda ax: histogram('Distribution of Churned Customers Tenure Months', 'Tenure Months', df_churn, 'Tenure Months', 20, ax=ax))

    print('--- Distribution of Churn Score of Churned Customers ---')
    plot_funcs.append(lambda ax: histogram('Distribution of Churned Customers Churn Score', 'Churn Score', df_churn, 'Churn Score', 20, ax=ax))

    print('--- Distribution of CLTV of Churned Customers ---')
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
        plt.show()

    return df

def bin_df(df, show=False):
    """
    Bin continuous columns into categorical groups and optionally display plots.

    This function bins 'Tenure Months', 'Churn Score', and 'CLTV' columns into
    categorical groups with meaningful labels. Optionally, it generates count
    plots of these binned columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing Telco customer data.
    show : bool, optional
        If True, generates and displays count plots for the binned columns. Default is False.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with new binned categorical columns added.

    Raises
    ------
    KeyError
        If required columns for binning are missing.
    """
    if show:
        plots_count = 3
        cols = 3
        rows = (plots_count + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axs = np.atleast_1d(axs).flatten()
    
    else:
        axs = [None, None, None]

    df = bin_and_plot('Distribution of Customers Grouped by Tenure Group', 'Tenure Group', df, 'Tenure Months', 'Tenure Group',
                      [0, 12, 30, 50, df['Tenure Months'].max()],
                      ['New Customer', 'New/Established Customer', 'Established/Veteran Customer', 'Veteran Customer'],
                      show_plot=show, ax=axs[0] if show else None)
    
    df = bin_and_plot('Distribution of Customers Grouped by Probability of Churn', 'Churn Probability', df, 'Churn Score',
                      'Churn Probability', [0, 25, 50, 75, 100],
                      ['Less Probability of Churn', 'Less/Moderate Probability of Churn', 'Moderate/High Probability of Churn', 'High Probability of Churn'],
                      tick_rotation=10, show_plot=show, ax=axs[1] if show else None)
    
    df = bin_and_plot('Distribution of Customers Grouped by Customer Value', 'Customer Value', df, 'CLTV', 'Customer Value',
                      [0, 3000, 4000, 5000, df['CLTV'].max()],
                      ['Low Value', 'Low/Mid Value', 'Mid/High Value', 'High Value'], show_plot=show, ax=axs[2] if show else None)
    
    if show is True:
        for ax in axs:
            ax.title.set_fontsize(9)
            ax.tick_params(axis='x', labelsize=6)
        
        plt.tight_layout()
        plt.show()

    return df


def hypothesis_test(df):
    """
    Perform chi-squared hypothesis test on 'Contract' vs 'Churn Value'.

    This function runs a chi-squared test to examine the independence
    between contract types and churn outcomes.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing binned Telco customer data.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If required columns are missing from the DataFrame.
    """
    _ , _ , _ = chi_squared_test(df, 'Contract', 'Churn Value')

def get_model(df):
    """
    Train and evaluate multiple classifiers, returning the best model and data splits.

    This function trains KNeighbors, RandomForest, and GaussianNB classifiers,
    evaluates them based on recall score, and returns the best performing model
    along with training/testing datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        The preprocessed and binned dataset.

    Returns
    -------
    tuple
        - model_results (dict): Dictionary containing evaluation results for models.
        - best_model (sklearn estimator): The best performing trained model.
        - X_train (pandas.DataFrame): Training feature set.
        - X_test (pandas.DataFrame): Testing feature set.
        - y_test (pandas.Series): Testing target variable.

    Raises
    ------
    ValueError
        If the dataset is invalid or model training fails.
    """
    _ ,model_results, best_model, X_train, X_test, y_test = comparative_models(
        df, 'Churn Value',{'KNeighbors': KNeighborsClassifier(), 'RandomForest': RandomForestClassifier(),'GaussianNB': GaussianNB()},
        ['State', 'City', 'Zip Code', 'Latitude', 'Longitude', 'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason',
         'Churn Probability', 'Customer Value', 'Tenure Months', 'Total Charges'], 'recall')
    
    return model_results, best_model, X_train, X_test, y_test
    
def global_explainer(model, X_train, X_test):
    """
    Generate global explainability visualizations for the given model.

    This function computes and displays global feature importances or
    model behavior explanations based on training and testing data.

    Parameters
    ----------
    model : sklearn estimator
        The trained classification model.
    X_train : pandas.DataFrame
        Training features.
    X_test : pandas.DataFrame
        Testing features.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If explanation generation fails.
    """
    model_global_explainer(model, X_train, X_test)

def local_explainer(model, X_train, X_test, index=None):
    """
    Generate local explainability visualization for a specific test sample.

    This function explains the prediction of the model for a single data point
    identified by `index` in the test set.

    Parameters
    ----------
    model : sklearn estimator
        The trained classification model.
    X_train : pandas.DataFrame
        Training features.
    X_test : pandas.DataFrame
        Testing features.
    index : int, optional
        The index of the test sample to explain. Defaults to 0.

    Returns
    -------
    None

    Raises
    ------
    IndexError
        If the index is out of bounds of the test dataset.
    Exception
        If explanation generation fails.
    """
    model_local_explainer(model, X_train, X_test, index)

def profit_curve_threshold(df, aggfunc, col, model_df, cost, retention_rate, y_test):
    """
    Calculate the optimal profit threshold from the profit curve analysis.

    This function computes the best threshold to maximize profit based on model
    predictions, a specified aggregation function on a column (e.g., median on CLTV),
    cost of retention, retention rate, and test target labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing customer data.
    aggfunc : str
        Aggregation function name to apply on column `col` (e.g., 'median', 'mean').
    col : str
        The column on which to apply the aggregation function.
    model_df : pandas.DataFrame
        DataFrame with model prediction results.
    cost : float
        Cost associated with retention effort per customer.
    retention_rate : float
        Expected retention rate after intervention.
    y_test : pandas.Series or array-like
        True target labels for the test dataset.

    Returns
    -------
    float
        Optimal threshold value to maximize profit.

    Raises
    ------
    ValueError
        If inputs are invalid or computation fails.
    """
    median_col = df_aggfunc(df, aggfunc, col)

    threshold, profit, _ = profit_curve(model_df, median_col, cost, retention_rate, y_test)
    print(f"Profit Curve Threshold: {threshold}, Profit: {profit}")

    return threshold

def deploy_model(df, model, target, cols_to_drop):
    """
    Deploy the selected model on the dataset, preparing for production use.

    This function runs the deployment pipeline by applying the model to the
    dataset after dropping specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to apply the model on.
    model : sklearn estimator
        The trained classification model.
    target : str
        Name of the target column in the dataset.
    cols_to_drop : list of str
        List of columns to drop before model deployment.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If columns to drop are missing from the DataFrame.
    Exception
        If deployment process fails.
    """
    deployment_model(df, model, target, cols_to_drop)

def generate_test_data():
    """
    Generate synthetic test data for model evaluation and prediction.

    This function creates a synthetic dataset with similar structure to the
    Telco churn dataset and applies binning on tenure groups.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Synthetic test dataset with binned features.

    Raises
    ------
    Exception
        If data generation or binning fails.
    """
    data = generate_data()
    data = bin_and_plot('', '', data, 'Tenure Months', 'Tenure Group', [0, 12, 30, 50, data['Tenure Months'].max()],
                        ['New Customer', 'New/Established Customer', 'Established/Veteran Customer', 'Veteran Customer'], show_plot=False)
    
    return data

def predict_df(df, threshold=0.5):
    """
    Predict churn on the given dataset using a specified probability threshold.

    This function applies the trained model to the dataset and assigns churn
    predictions based on the given threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset for churn prediction.
    threshold : float, optional
        Probability threshold for classifying churn. Default is 0.5.

    Returns
    -------
    pandas.DataFrame
        DataFrame with churn prediction columns added.

    Raises
    ------
    ValueError
        If threshold is outside valid range [0,1].
    """
    predicted_df = predict_churn(df, threshold)

    return predicted_df

def abc_test_assignment(df):
    """
    Assign ABC test groups based on predicted churn probabilities.

    This function segments the dataset into groups (A, B, C) for targeted
    retention testing based on predicted churn scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with churn predictions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ABC test assignment column added.

    Raises
    ------
    KeyError
        If required prediction columns are missing.
    """
    data = abc_test(df)

    return data