"""Console script for telco_customer_churn_analysis."""

import typer
from rich.console import Console
import joblib
from typing import Optional

from .telco_customer_churn_analysis import (data_preprocessing, exploratory_analysis,profit_curve_threshold, deploy_model,
                                            get_model,abc_test_assignment, predict_df, bin_df, generate_test_data, hypothesis_test,
                                            local_explainer, global_explainer, features_to_dataframe)

app = typer.Typer()
console = Console()


@app.command()
def eda():
    """Run exploratory data analysis (EDA) on the Telco Customer dataset."""
    df = data_preprocessing()
    df = exploratory_analysis(df)


@app.command()
def hypothesis_tests_chi2(test_or_new='Test'):
    """
    Performs Chi-Squared hypothesis tests on the Telco dataset.
    
    Parameters
    ----------
    test_or_new : str, optimal
        Use 'test' for the preprocessed training dataset or 'New' to generate synthetic test data."""
    if test_or_new == 'Test':
        df = data_preprocessing()
        df = bin_df(df)
    elif test_or_new == 'New':
        df = generate_test_data()
        df = bin_df(df)
    else:
        print('Need to choose between using the training data (Test) or new generated data (New).')

    hypothesis_test(df)


@app.command()
def train_evaluate_deploy():
    """Train, evaluate, and deploy a classification model for customer churn."""
    df = data_preprocessing()
    df = bin_df(df)
    get_model(df)

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
                                       retention_rate: float = 0.8,
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
    Predict churn using the best threshold based on the profit curve.

    Parameters:
    ----------
    df : pd.DataFrame, optional
        Input data. If None, new synthetic data will be generated.
    y_test : pd.Series, optional
        True labels. If None, loaded from the saved model bundle.
    aggfunc : str, default='median'
        Aggregation function for profit calculation (e.g., 'mean', 'median').
    col : str, default='CLTV'
        Column used for customer value.
    cost : float, default=100.0
        Cost to retain a customer.
    retention_rate : float, default=0.8
        Expected retention success rate.
    City : str, optional
        Customer city.
    Gender : str, optional
        Customer gender.
    Senior_Citizen : str, optional
        Whether the customer is a senior citizen.
    Partner : str, optional
        Whether the customer has a partner.
    Dependents : str, optional
        Whether the customer has dependents.
    Tenure_Months : int, optional
        Number of months the customer has been with the company.
    Phone_Service : str, optional
        Whether the customer has phone service.
    Multiple_Lines : str, optional
        Whether the customer has multiple phone lines.
    Internet_Service : str, optional
        Type of internet service.
    Online_Security : str, optional
        Whether the customer has online security service.
    Online_Backup : str, optional
        Whether the customer has online backup service.
    Device_Protection : str, optional
        Whether the customer has device protection service.
    Tech_Support : str, optional
        Whether the customer has tech support service.
    Streaming_TV : str, optional
        Whether the customer streams TV.
    Streaming_Movies : str, optional
        Whether the customer streams movies.
    Contract : str, optional
        Customer contract type.
    Paperless_Billing : str, optional
        Whether the customer uses paperless billing.
    Payment_Method : str, optional
        Customer payment method.
    Monthly_Charges : float, optional
        Monthly charges for the customer.
    Total_Charges : float, optional
        Total charges for the customer.
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

    df = bin_df(df)

    if y_test is None:
        with open('model_results.pkl', 'rb') as deployed_model:
            bundle = joblib.load(deployed_model)
        
        y_test = bundle['y_test']

    with open('model_results.pkl', 'rb') as deployed_model:
        bundle = joblib.load(deployed_model)
    
    model_df = bundle['all_results']
    
    threshold = profit_curve_threshold(df, aggfunc, col, model_df, cost, retention_rate, y_test)

    predicted_df = predict_df(df, threshold)
    print('Model deployed successfully!')
    print(f'Best threshold from profit curve: {threshold}')
    
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
    """
    Predict churn and optionally explain results using global or local XAI.

    Parameters:
    ----------
    df : pd.DataFrame, optional
        Input data. If None, synthetic test data will be generated.
    threshold_input : float, default=0.5
        Probability threshold for churn classification.
    --global-xai : flag
        Enable SHAP global explanation.
    --local-xai : flag
        Enable SHAP local explanation for a specific customer.
    index_local : int, default=0
        Row index for the customer explanation in local XAI.
    City : str, optional
        Customer city.
    Gender : str, optional
        Customer gender.
    Senior_Citizen : str, optional
        Whether the customer is a senior citizen.
    Partner : str, optional
        Whether the customer has a partner.
    Dependents : str, optional
        Whether the customer has dependents.
    Tenure_Months : int, optional
        Number of months the customer has been with the company.
    Phone_Service : str, optional
        Whether the customer has phone service.
    Multiple_Lines : str, optional
        Whether the customer has multiple phone lines.
    Internet_Service : str, optional
        Type of internet service.
    Online_Security : str, optional
        Whether the customer has online security service.
    Online_Backup : str, optional
        Whether the customer has online backup service.
    Device_Protection : str, optional
        Whether the customer has device protection service.
    Tech_Support : str, optional
        Whether the customer has tech support service.
    Streaming_TV : str, optional
        Whether the customer streams TV.
    Streaming_Movies : str, optional
        Whether the customer streams movies.
    Contract : str, optional
        Customer contract type.
    Paperless_Billing : str, optional
        Whether the customer uses paperless billing.
    Payment_Method : str, optional
        Customer payment method.
    Monthly_Charges : float, optional
        Monthly charges for the customer.
    Total_Charges : float, optional
        Total charges for the customer.
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
    
    df = bin_df(df)

    predicted_df = predict_df(df, threshold_input)
    abc_df = abc_test_assignment(predicted_df)

    print('Prediction Results with ABC assignment')
    print(abc_df)

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
