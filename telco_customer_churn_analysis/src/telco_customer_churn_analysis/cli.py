"""Console script for telco_customer_churn_analysis."""

import typer
from rich.console import Console
import joblib
from typing import Optional

from src.telco_customer_churn_analysis.telco_customer_churn_analysis import (data_preprocessing, exploratory_analysis,profit_curve_threshold, deploy_model,
                                            get_model,abc_test_assignment, predict_df, bin_df, generate_test_data, hypothesis_test,
                                            local_explainer, global_explainer, features_to_dataframe)
from src.telco_customer_churn_analysis.telco_customer_churn_analysis_app import bin_df_app
app = typer.Typer()
console = Console()


@app.command()
def eda():
    """Run exploratory data analysis (EDA) on the Telco Customer dataset."""
    df = data_preprocessing()
    df = exploratory_analysis(df)


@app.command()
def hypothesis_tests_chi2(data_choice: str = 'Test', col1: str = None, col2: str = None):
    """
    Perform a Chi-squared test of independence between two categorical variables.

    Depending on the `data_choice` parameter, this function either uses the 
    preprocessed Telco test dataset or generates new synthetic data. It bins 
    the data before performing the Chi-squared hypothesis test.

    Parameters
    ----------
    data_choice : str, default='Test'
        Dataset selection:
        - 'Test': Use preprocessed Telco test data.
        - 'New' : Generate and use new synthetic data.
    col1 : str, optional
        Name of the first column to test. If None or invalid, defaults to the first column in the DataFrame.
    col2 : str, optional
        Name of the second column to test. If None or invalid, defaults to the second column in the DataFrame.

    Raises
    ------
    ValueError
        If `data_choice` is invalid or if the DataFrame has insufficient columns for testing.
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
    
    hypothesis_test(data_choice, col1, col2)


@app.command()
def train_evaluate_deploy():
    """Train, evaluate, and deploy a classification model for customer churn."""
    df = data_preprocessing()
    df = bin_df(df)
    print('Get Model')
    get_model(df)

    print('Cli Train before model')

    with open('model_results.pkl', 'rb') as r:
        model_results = joblib.load(r)

    best_model = model_results['model_untrained']

    deploy_model(
        df, best_model, target='Churn Value',
        cols_to_drop=[
            'State', 'Zip Code', 'Latitude', 'Longitude','Churn Value', 'Churn Score', 'CLTV', 'Churn Reason',
            'Churn Probability', 'Customer Value', 'Tenure Months','Total Charges']
            )


@app.command()
def predict_with_best_profit_threshold(df=None, y_test=None, aggfunc: str = 'median', col: str = 'CLTV', cost: float = 100.0,
                                       retention_rate: float = 0.8, abc_assignment: bool = False,
                                       City: Optional[str] = None, Gender: Optional[str] = None, Senior_Citizen: Optional[str] = None,
                                       Partner: Optional[str] = None, Dependents: Optional[str] = None, Tenure_Months: Optional[int] = None,
                                       Phone_Service: Optional[str] = None, Multiple_Lines: Optional[str] = None,
                                       Internet_Service: Optional[str] = None, Online_Security: Optional[str] = None,
                                       Online_Backup: Optional[str] = None, Device_Protection: Optional[str] = None,
                                       Tech_Support: Optional[str] = None, Streaming_TV: Optional[str] = None,
                                       Streaming_Movies: Optional[str] = None, Contract: Optional[str] = None,
                                       Paperless_Billing: Optional[str] = None, Payment_Method: Optional[str] = None,
                                       Monthly_Charges: Optional[float] = None, Total_Charges: Optional[float] = None):
    """Predict churn using the threshold that maximizes profit."""
    
    """
    This function predicts customer churn using the threshold derived from the profit curve.
    Optional customer features can be provided as input. If none are given, synthetic test data is generated.
    The function can also perform ABC assignment if requested.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input data. If None, synthetic data will be generated.
    y_test : pd.Series, optional
        True labels for profit calculation. If None, loaded from saved model bundle.
    aggfunc : str, default='median'
        Aggregation function for profit calculation.
    col : str, default='CLTV'
        Column representing customer value.
    cost : float, default=100.0
        Cost to retain a customer.
    retention_rate : float, default=0.8
        Expected retention success rate.
    abc_assignment : bool, default=False
        Whether to perform ABC assignment.
    City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
    Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
    Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges : optional
        Individual customer features.
    """
    features_provided = any([City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
                             Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
                             Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges])
    
    if features_provided:
        df = features_to_dataframe(City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
                        Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
                        Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges)
    
    elif df is not None:
        df = df.copy()
    
    else:
        df = generate_test_data()

    df = bin_df_app(df)

    print('--- User Df ---')
        
    try: 
        with open('model_results.pkl', 'rb') as deployed_model:
            bundle = joblib.load(deployed_model)

    except (EOFError, FileNotFoundError, KeyError, joblib.externals.loky.process_executor.TerminatedWorkerError) as e:
        print(f"Warning: Could not load 'model_results.pkl' ({type(e).__name__} - {e}).")
        print('Please run `comparative_models` to generate a valid model bundle.')
            
        return
        
    except Exception as e:
        print(f"Unexpected error loading 'model_results' ({type(e).__name__} - {e}).")
            
        return

    if y_test is None:
        y_test = bundle['y_test']
    
    model_df = bundle['all_results']
    
    threshold = profit_curve_threshold(aggfunc, col, model_df, cost, retention_rate, y_test)
    
    print('--- Predict ---')
    predicted_df = predict_df(df, threshold)
    print('Model deployed successfully!')
    print(f'Best threshold from profit curve: {threshold}')
    
    if abc_assignment:
        abc_df = abc_test_assignment(predicted_df)
        print('Prediction Results with ABC assignment')
        print(abc_df)


@app.command()
def predict_with_xai(df = None, threshold_input: float = 0.5,
                     global_xai: bool = typer.Option(False, "--global-xai", help="Enable Global XAI"),
                     local_xai: bool = typer.Option(False, "--local-xai", help="Enable Local XAI"),
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
    """Predict churn and optionally explain with global or local XAI."""
    
    """
    This function predicts customer churn and can generate SHAP explanations:
    - Global XAI: explains the overall model behavior.
    - Local XAI: explains a specific customer's prediction.

    Optional customer features can be provided. If none are given, synthetic test data is generated.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input data. If None, synthetic data will be generated.
    threshold_input : float, default=0.5
        Probability threshold for churn classification.
    global_xai : bool, default=False
        Enable global SHAP explanations.
    local_xai : bool, default=False
        Enable local SHAP explanations for a single customer.
    index_local : int, default=0
        Row index for local explanation.
    City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
    Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
    Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges : optional
        Individual customer features.
    """
    
    features_provided = any([City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
                             Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
                             Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges])
    
    if features_provided:
        df = features_to_dataframe(City, Gender, Senior_Citizen, Partner, Dependents, Tenure_Months, Phone_Service, Multiple_Lines,
                        Internet_Service, Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV,
                        Streaming_Movies, Contract, Paperless_Billing, Payment_Method, Monthly_Charges, Total_Charges)
    elif df is not None:
        df = df.copy()

    else:
        df = generate_test_data()
    
    df = bin_df_app(df)

    predicted_df = predict_df(df, threshold_input)

    with open('deployment_pipeline.pkl', 'rb') as deployed_model:
        bundle = joblib.load(deployed_model)
        
    model = bundle

    if global_xai is True:
        with open('model_results.pkl', 'rb') as deployed_model:
            bundle = joblib.load(deployed_model)

        X_train, X_test = bundle['X_train'], bundle['X_test']

        global_explainer(model, X_train, X_test)

    if local_xai is True:
        with open('model_results.pkl', 'rb') as deployed_model:
            bundle = joblib.load(deployed_model)

        X_train, X_test = bundle['X_train'], bundle['X_test']

        local_explainer(model, X_train, X_test, index_local)



if __name__ == "__main__":
    app()
