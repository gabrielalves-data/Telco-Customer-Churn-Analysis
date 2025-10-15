import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve ,roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from itertools import product
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from typing import Union, Dict, List, Any, Optional, Tuple
import joblib

from src.telco_customer_churn_analysis.utils import (safe_display)

def preprocess_data(df: pd.DataFrame, target: str,
                    cols_to_drop: Optional[Union[str, List[str]]] = None, test_size: float = 0.2,
                    random_state: int = 123) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, ColumnTransformer]:
    """
    Prepares data for machine learning by splitting and defining the preprocessing logic.

    This function performs key data preparation steps, including splitting, feature
    selection based on cardinality and type, and defining a ColumnTransformer
    for preprocessing.

    Preprocessing Logic:
    - One-Hot Encoding ('one_hot'): Applied to categorical/object columns with 5 or fewer unique values.
    - Standard Scaling ('scaler'): Applied to numerical columns (int64/float64) with more than 5 unique values.
    - Remainder ('passthrough'): All other columns (including high-cardinality categoricals and
      low-cardinality numerics) are passed through unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    target : str
        The name of the target variable column.
    cols_to_drop : str or list of str, optional
        Additional columns to drop *besides* the target. Defaults to None.
    test_size : float, optional
        The proportion of the dataset to include in the test split (0 to 1). Defaults to 0.2.
    random_state : int, optional
        The seed used by the random number generator. Defaults to 123.

    Returns
    -------
    tuple: A tuple containing:
        - X (pandas.DataFrame): The full features DataFrame.
        - y (pandas.Series): The full target Series.
        - X_train (pandas.DataFrame): The training features.
        - X_test (pandas.DataFrame): The testing features.
        - y_train (pandas.Series): The training target.
        - y_test (pandas.Series): The testing target.
        - preprocessor (sklearn.compose.ColumnTransformer): The **unfitted** ColumnTransformer object.
        - preprocessor_trained (sklearn.compose.ColumnTransformer): The **fitted** ColumnTransformer object (fitted only on X_train).

    Raises
    ------
    TypeError
        If inputs are of the wrong type (e.g., df is not a DataFrame).
    KeyError
        If `target` or any column in `cols_to_drop` is not found in the DataFrame.
    ValueError
        If `test_size` is outside the (0, 1) range.
    RuntimeError
        For unexpected issues during processing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"TypeError: Input 'df' must be a pandas DataFrame, but received {type(df).__name__}.")

    if not isinstance(target, str):
        raise TypeError(f"TypeError: 'target' must be a string.")

    if target not in df.columns:
        raise KeyError(f"KeyError: Target column '{target}' not found in DataFrame. Available columns {list(df.columns)}.")

    if not isinstance(test_size, float) or not (0 < test_size < 1):
        raise ValueError(f"ValueError: 'test_size'must be a float between 0 and 1.")

    if not isinstance(random_state, int):
        raise TypeError(f"TypeError: 'random_state' must be an integer.")

    if cols_to_drop is None:
        columns_to_drop = [target]

    elif isinstance(cols_to_drop, str):
        columns_to_drop = [target, cols_to_drop]

    elif isinstance(cols_to_drop, list):
        columns_to_drop = [target] + cols_to_drop

    else:
        raise TypeError(f"TypeError: 'cols_to_drop' must be a string, a list of strings, or None.")

    for col in columns_to_drop:
        if col != target and col not in df.columns:
            raise KeyError(f"KeyError: Column to drop '{col}' not found in DataFrame. Available columns {list(df.columns)}.")

    try:
        X = df.drop(columns_to_drop, axis=1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        one_hot_cols = [col for col in X.columns if X[col].nunique() <= 5 and X[col].dtype in ['object', 'category', 'bool']]

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

        scaler_cols = [col for col in numeric_cols if X[col].nunique() > 5]


        preprocessor = ColumnTransformer(
            transformers=[
                ('one_hot', OneHotEncoder(handle_unknown='ignore'), one_hot_cols),
                ('scaler', StandardScaler(), scaler_cols)
            ],
            remainder='passthrough'
        )

        preprocessor_trained = preprocessor.fit(X_train)

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during data splitting or transformer definition. Details: {e}.")

    return X, y, X_train, X_test, y_train, y_test, preprocessor, preprocessor_trained


def model_pipelines(preprocessor: ColumnTransformer, comparative_models: Dict[str, BaseEstimator],
                    random_state: int = 123) -> Dict[str, Pipeline]:
    """
    Creates a dictionary of scikit-learn pipelines for different machine learning models.

    This function streamlines the machine learning workflow by combining a
    preprocessing step with various scikit-learn classifiers into a single pipeline.
    It automatically includes a Logistic Regression pipeline (the baseline) and
    then creates a pipeline for each model provided in the `comparative_models` dictionary.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        The unfitted preprocessor object (ColumnTransformer) to be used as the first step
        in all pipelines.
    comparative_models : dict
        A dictionary where keys are model names (str) and values are initialized model objects
        (sklearn estimators).
    random_state : int, optional
        The random state to set for all instantiated models that support it, ensuring reproducibility.
        Defaults to 123.

    Returns
    -------
    dict
        A dictionary of scikit-learn Pipeline objects, with keys derived from the model names.

    Raises
    ------
    TypeError
        If inputs are of the wrong type (e.g., preprocessor is not a ColumnTransformer).
    ValueError
        If the `comparative_models` dictionary is empty.
    RuntimeError
        For unexpected issues during pipeline creation.
    """

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(f"TypeError: 'preprocessor' must be an sklearn ColumnTransformer, but received {type(preprocessor).__name__}.")

    if not isinstance(comparative_models, dict):
        raise TypeError(f"TypeError: 'comparative_models' must be a dictionary, but received {type(comparative_models).__name__}.")

    if not comparative_models:
        raise ValueError(f"ValueError: 'comparative_models' dictionary cannot be empty.")

    if not isinstance(random_state, int):
        raise TypeError(f"TypeError: 'random_state' must be an integer.")

    models: Dict[str, Pipeline] = {}

    try:
        models['LogisticRegression'] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=random_state, solver='liblinear'))
        ])

        for name, model in comparative_models.items():
            if not isinstance(model, BaseEstimator):
                raise TypeError(f"Invalid model type for '{name}'. Expected an sklearn estimator but got {type(model).__name__}.")
            if hasattr(model, 'random_state'):
                model.set_params(random_state=random_state)

            pipeline_name = name.replace('Classifier', '').strip(' ')

            models[pipeline_name] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

    except Exception as e:
        raise RuntimeError(f"RuntimeError: Failed to create one or more pipelines. Ensure all items in 'comparative_models' are valid sklearn estimators. Details: {e}.")

    return models


def hyperparameters(models: Dict[str, Pipeline]) -> Dict[str, Dict[str, Union[Pipeline, Dict[str, List[Any]]]]]:
    """
    Defines hyperparameter grids for different machine learning models within pipelines.

    This function iterates through a dictionary of scikit-learn pipelines and,
    based on the type of classifier in each pipeline, creates a dictionary of
    hyperparameter grids suitable for grid search or randomized search.
    Each model's entry includes the pipeline object and a 'params' dictionary
    with parameter names correctly prefixed by 'classifier__'.

    Parameters
    ----------
    models : dict
        A dictionary of scikit-learn Pipeline objects, where each
        pipeline's final step is a classifier.

    Returns
    -------
    dict
        A dictionary where keys are model names and values are dictionaries
        containing the pipeline object ('model') and a corresponding
        hyperparameter grid ('params'). Models without a defined grid will be
        omitted, and a message will be printed.

    Raises
    ------
    TypeError
        If `models` is not a dictionary.
    ValueError
        If `models` is empty or contains items that are not scikit-learn Pipelines.
    RuntimeError
        For unexpected issues during processing.
    """

    if not isinstance(models, dict):
        raise TypeError(f"TypeError: 'models' must be a dictionary, but received {type(models).__name__}.")

    if not models:
        raise ValueError(f"ValueError: The 'models' dictionary cannot be empty.")


    params: Dict[str, Union[Pipeline, Dict[str, List[Any]]]] = {}

    try:
        for name, pipeline in models.items():
            if not isinstance(pipeline, Pipeline):
                print(f"Warning: Skipping '{name}'. Value is not a scikit-learn Pipeline.")
                continue

            model = pipeline.named_steps['classifier']

            if isinstance(model, LogisticRegression):
                params[name] = {
                    'model': pipeline,
                    'params': {
                        'classifier__penalty': ['l1', 'l2'],
                        'classifier__C': [0.1, 0.5, 1, 5],
                        'classifier__solver': ['liblinear'],
                        'classifier__class_weight': ['balanced', None]
                    }
                }

            elif isinstance(model, KNeighborsClassifier):
                params[name] = {
                    'model': pipeline,
                    'params': {
                        'classifier__n_neighbors': [1,2,3,4],
                        'classifier__weights': ['uniform', 'distance'],
                        'classifier__p': [1, 2]
                    }
                }

            elif isinstance(model, RandomForestClassifier):
                params[name] = {
                    'model': pipeline,
                    'params': {
                        'classifier__n_estimators': [15, 20, 25, 30],
                        'classifier__max_depth': [2,5,10,15],
                        'classifier__min_samples_leaf': [5,10,15],
                        'classifier__max_features': ['sqrt', 'log2', None],
                        'classifier__class_weight': ['balanced', None]
                    }
                }

            elif isinstance(model, GaussianNB):
                params[name] = {
                    'model': pipeline,
                    'params': {
                        'classifier__var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]
                    }
                }

            else:
                print(f'Hyperparameters for {name} not defined in this function and will be ommitted from tuning.')

    except Exception as e:
        raise RuntimeError(f"RuntimeError: An unexpected error occurred during parameter grid definition: {e}.")

    return params


def train_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                         y_test: pd.Series,
                         params: Dict[str, Dict[str, Union[Pipeline, Dict[str, List[Any]]]]],
                         metric: str = 'f1', cv: int = 5) -> Tuple[Dict[str, Pipeline], List[Dict[str, Any]]]:
    """
    Trains and evaluates multiple machine learning models using GridSearchCV.

    This function iterates through a dictionary of models and their hyperparameter
    grids. For each model, it performs a grid search with cross-validation to
    find the best hyperparameters. It then trains the best estimator, makes
    predictions on the test set, and calculates and prints various
    classification metrics (Recall, Precision, F1-Score, Accuracy, and ROC-AUC).
    It also identifies and stores the top 4 most important features.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame
        The training and testing feature sets.
    y_train, y_test : pandas.Series
        The training and testing target variables.
    params : dict
        A dictionary containing model pipelines and their corresponding
        hyperparameter grids, typically generated by the `hyperparameters` function.
    metric : str, optional
        The scoring metric to use for GridSearchCV. Defaults to 'f1'.
    cv : int, optional
        The number of cross-validation folds (k-folds) to use for GridSearchCV.
        Defaults to 5.

    Returns
    -------
    tuple
        - best_models (dict): A dictionary of the best trained estimator
          (the final GridSearchCV.best_estimator_) for each model.
        - model_results (list): A list of dictionaries, where each
          dictionary contains the evaluation results, best parameters,
          predictions, and prediction probabilities for a single model.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input data or parameter structures are invalid.
    RuntimeError
        For unexpected errors during model training or evaluation.
    """

    if not all(isinstance(df, (pd.DataFrame, pd.Series)) for df in [X_train, X_test, y_train, y_test]):
        raise TypeError(f"TypeError: Input data X_train, X_test, y_train, and y_test must be pandas DataFrame or Series.")

    if not isinstance(params, dict) or not params:
        raise ValueError(f"The 'params' dictionary must be a non-empty dictionary.")

    if not isinstance(cv, int) or cv <= 1:
        raise ValueError(f"ValueError: 'cv' must be an integer greater than 1.")

    best_models: Dict[str, Pipeline] = {}
    model_results: List[Dict[str, Any]] = []

    for name, setup in params.items():
        try:
            if not isinstance(setup, dict) or 'model' not in setup or 'params' not in setup:
                print(f"Warning: Skipping '{name}'. Setup is invalid or missing 'model' or 'params' keys.")
                continue

            grid_search = GridSearchCV(
                estimator=setup['model'],
                param_grid=setup['params'],
                cv=cv,
                scoring=metric,
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test)[:,1]
                _roc_auc_score = roc_auc_score(y_test, y_proba)
            else:
                y_proba = None
                _roc_auc_score = 'N/A'


            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            _f1_score = f1_score(y_test, y_pred)


            feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
            classifier = best_model.named_steps['classifier']
            top_features: Union[Dict[str, float], str] = {}


            if hasattr(classifier, 'feature_importances_'):
                importances = pd.Series(classifier.feature_importances_, index=feature_names)
                top_features = importances.sort_values(ascending = False).head(4).to_dict()

            elif hasattr(classifier, 'coef_'):
                if len(classifier.coef_.shape) > 1 and classifier.coef_.shape[0] > 1:
                    coefficients = pd.Series(classifier.coef_[0], index=feature_names)
                else:
                    coefficients = pd.Series(classifier.coef_.flatten(), index=feature_names)

                top_features = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index).head(4).to_dict()

            else:
                top_features = f'Feature importance is not available for {type(classifier).__name__}.'


            print(f'Model Name: {name}')
            print(f'Recall: {recall:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1-Score: {_f1_score:.4f}')

            if _roc_auc_score != 'N/A':
                print(f'Roc-Auc-Score: {_roc_auc_score:.4f}')
            else:
                print('Roc-Auc-Score: N/A')

            print(f'Feature Importance: {top_features}\n')


            model_results.append({'name': name ,'params': grid_search.best_params_, 'recall': recall,
                                  'precision': precision, 'f1': _f1_score, 'roc_auc': _roc_auc_score,
                                  'accuracy': accuracy, 'top_features': top_features,
                                  'predictions': y_pred, 'predictions_proba': y_proba})

            best_models[name] = best_model

        except Exception as e:
            print(f"Error occurred while training/evaluating '{name}': {e}.")
            model_results.append({'name': name})

    return best_models, model_results


def voting_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                 best_models: Dict[str, Pipeline], metric: str = 'f1', n_iter: int = 10,
                 cv: int = 5, random_state: int = 123) -> Tuple[Dict[str, VotingClassifier], List[Dict[str, Any]]]:
    """
    Trains and evaluates a VotingClassifier using a combination of best models.

    This function creates an ensemble VotingClassifier using a 'soft' voting
    strategy (aggregating probabilities) from the base models provided. It dynamically
    constructs a hyperparameter search space for the combined model by generating a
    set of discrete weight combinations (using `itertools.product` internally) and then uses
    RandomizedSearchCV to sample from these combinations to find the optimal set of
    base model weights. Finally, it evaluates and prints the performance metrics of the best
    VotingClassifier on the test set.

    Parameters
    ----------
    X_train (pandas.DataFrame): The training feature set.
    X_test (pandas.DataFrame): The testing feature set.
    y_train (pandas.Series): The training target variable.
    y_test (pandas.Series): The testing target variable.
    best_models (dict): A dictionary of best-performing trained models,
        typically from a previous grid search. Keys are model names, values are Pipeline objects.
    metric (str, optional): The scoring metric to use for RandomizedSearchCV.
        Defaults to 'f1'.
    n_iter (int, optional): The number of parameter settings that are sampled
        during RandomizedSearchCV. Defaults to 10.
    cv (int, optional): The number of cross-validation folds (k-folds)
        to use for RandomizedSearchCV. Defaults to 5.
    random_state (int, optional): The seed for the random number generator
        for reproducibility. Defaults to 123.

    Returns
    -------
    tuple
        - best_voting_model (dict): A dictionary containing the best trained
          VotingClassifier object under the key 'VotingClassifier'.
        - vot_model_results (list): A list containing a dictionary of the
          evaluation results (including accuracy, recall, precision, F1, AUC,
          best parameters), predictions, and prediction probabilities for the VotingClassifier.
    """

    if not isinstance(best_models, dict) or not best_models:
        print("Error: 'best_models' dictionary is empty or invalid. Cannot create a Voting Classifier.")
        return {}, []

    estimators = [(name, model) for name, model in best_models.items()]
    num_estimators = len(estimators)

    if any(not hasattr(model, 'predict_proba') for _, model in estimators):
        print("Warning: One or more base models do not support 'predict_proba'. VotingClassifier may fall back to 'hard' voting.")

    voting_model = VotingClassifier(estimators=estimators, voting='soft')

    best_voting_model: Dict[str, VotingClassifier] = {}
    vot_model_results: List[Dict[str, Any]] = []

    try:
        weight_options = [1, 2, 3]

        all_combinations = list(product(weight_options, repeat=num_estimators))

        weight_list = [list(c) for c in all_combinations if len(set(c)) > 1]
        weight_list.append(None)
        voting_params = {'weights': weight_list}

        rand_search_vot = RandomizedSearchCV(
            estimator = voting_model,
            param_distributions={'weights': voting_params['weights']},
            n_iter=n_iter,
            cv=cv,
            scoring=metric,
            n_jobs=-1,
            verbose=0,
            random_state=random_state
            )

        rand_search_vot.fit(X_train, y_train)

        best_vot_model = rand_search_vot.best_estimator_
        best_params_vot = rand_search_vot.best_params_

        y_pred_vot = best_vot_model.predict(X_test)
        y_proba_vot = best_vot_model.predict_proba(X_test)[:,1]

        accuracy_vot = accuracy_score(y_test, y_pred_vot)
        precision_vot = precision_score(y_test, y_pred_vot)
        recall_vot = recall_score(y_test, y_pred_vot)
        f1_vot = f1_score(y_test, y_pred_vot)
        auc_vot = roc_auc_score(y_test, y_proba_vot)

        print('-----Voting Classifier Statistics-----')
        print(f'Recall: {recall_vot:.4f}')
        print(f'Precision: {precision_vot:.4f}')
        print(f'Accuracy: {accuracy_vot:.4f}')
        print(f'F1: {f1_vot:.4f}')

        if auc_vot != 'N/A':
            print(f'AUC: {auc_vot:.4f}')

        else:
            print('AUC: N/A')

        best_voting_model['VotingClassifier'] = best_vot_model

        vot_model_results.append({'name': 'VotingClassifier' ,'params': rand_search_vot.best_params_,
                                  'recall': recall_vot,'precision': precision_vot,
                                  'f1': f1_vot, 'roc_auc': auc_vot, 'accuracy': accuracy_vot,
                                  'predictions': y_pred_vot, 'predictions_proba': y_proba_vot})

    except Exception as e:
        print(f"Error: A critical error occurred during VotingClassfier training. {e}")
        return best_voting_model, vot_model_results

    return best_voting_model, vot_model_results



def comparative_models(df: pd.DataFrame, target: str, comparative_models: Dict[str, ClassifierMixin],
                       cols_to_drop: Union[str, List[str], None] = None, metric: str = 'f1',
                       test_size: float = 0.2, random_state: int = 123) -> Tuple[Dict[str, Pipeline], pd.DataFrame, Pipeline, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Orchestrates a comprehensive machine learning workflow to train, evaluate, and compare
    multiple classification models, including a final voting ensemble.

    The process includes data preprocessing, baseline model establishment via cross-validation,
    hyperparameter tuning via grid search, and final evaluation of individual and
    ensemble models. The best model, based on the specified metric, is then identified
    and its details are printed. The function prints the baseline score, comparative
    model results, voting model results, and a summary table.

    NOTE: This function relies on several external helper functions (preprocess_data,
    model_pipelines, hyperparameters, train_evaluate_model, voting_model) which must
    be imported or defined in the execution environment.

    Parameters
    ----------
    df (pandas.DataFrame): The input DataFrame.
    target (str): The name of the target variable column.
    comparative_models (dict): A dictionary of scikit-learn classifier objects
        to compare against Logistic Regression.
    cols_to_drop (str or list, optional): Additional columns to drop from the
        features. Defaults to None.
    metric (str, optional): The scoring metric for model evaluation (used in GridSearchCV
        and for final sorting). Defaults to 'f1'.
    test_size (float, optional): The proportion of the dataset to include in the
        test split. Defaults to 0.2.
    random_state (int, optional): The seed for reproducibility. Defaults to 123.

    Returns
    -------
    tuple
        - all_models (dict): A dictionary containing all best-performing trained
          models (from GridSearch/RandomizedSearch), including the final VotingClassifier ensemble.
        - all_results (pandas.DataFrame): A DataFrame summarizing the evaluation
          metrics and best parameters for all trained models, sorted by the specified metric.
        - model_object_by_metric (sklearn.pipeline.Pipeline): The single best-performing
          trained Pipeline object, determined by the specified `metric`.
        - X_train (pandas.DataFrame): The training features used for model fitting.
        - X_test (pandas.DataFrame): The testing features used for final evaluation.
        - y_test (pandas.Series): The testing target variable.

    Raises
    ------
    ValueError, TypeError, RuntimeError
        For various issues during the workflow.
    """

    all_models = {}
    all_results = pd.DataFrame()
    model_object_by_metric: Union[Pipeline, None] = None
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_test = pd.Series(dtype='int')
    best_model_name = 'N/A'

    try:
        X, y, X_train, X_test, y_train, y_test, preprocessor, preprocessor_trained = preprocess_data(df, target, cols_to_drop, test_size, random_state)

        models = model_pipelines(preprocessor, comparative_models, random_state)
        params = hyperparameters(models)

        baseline_score = cross_val_score(models['LogisticRegression'], X, y, cv=5, scoring=metric)
        print('-' * 100)
        print(f'\nBaseline Logistic Regression Score: {np.mean(baseline_score):.4f}.\n')
        print('-' * 100)

        print('\n','-' * 50,'Comparative Models Results','-' * 50,'\n')
        best_models, model_results = train_evaluate_model(X_train, X_test, y_train, y_test, params, metric)

        best_voting_model, vot_model_results = voting_model(X_train, X_test, y_train, y_test, best_models,
                                                            metric=metric, random_state=random_state)

        model_results = pd.DataFrame(model_results)
        vot_model_results = pd.DataFrame(vot_model_results)

        all_results = pd.concat([model_results, vot_model_results], axis=0)
        best_models.update(best_voting_model)
        all_models = best_models

        if metric in all_results.columns:
            sort_cols = [metric]
            if metric != 'f1' and 'f1' in all_results.columns:
                sort_cols.append('f1')

            valid_sort_cols = [col for col in sort_cols if col in all_results.columns]
            if valid_sort_cols:
                all_results = all_results.sort_values(by=valid_sort_cols, ascending=False,
                                                      ignore_index=True)

        if not all_results.empty:
            best_model_name = all_results['name'].iloc[0]
            model_object_by_metric = all_models[best_model_name]

        else:
            best_model_name = 'N/A'
            model_object_by_metric = None

        print('\n','-' * 50,'Model Results','-' * 50,'\n')
        safe_display(all_results[['name', 'accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'params']])

        print('\n','-' * 50,f'Best Model: {best_model_name} by Metric: {metric}','-' * 50,'\n')

        return all_models, all_results, model_object_by_metric, X_train, X_test, y_test

    except NotImplementedError as e:
        print(f"Execution failed: A required helper function is missing: {e}.")
        return {}, pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int')

    except Exception as e:
        print(f"An unexpected error occurred during the workflow: {type(e).__name__} - {e}.")
        return {}, pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int')
    

## MLOPS, Optimization and Business Action Functions


def profit_curve(median_value: float, cost: float, retention_rate: float,
                 labels: Union[np.ndarray, pd.Series], predictions_proba: Union[np.ndarray, pd.Series]) -> Tuple[float, float, Dict[float, float]]:
    """
    Calculates the optimal prediction threshold and maximum expected profit
    for a classification model based on a defined cost/benefit matrix.

    This function simulates the financial outcome (profit or loss) of applying
    an intervention (e.g., a retention offer) to customers identified as high-risk
    across a fine-grained range of probability thresholds. It uses a cost-sensitive
    evaluation defined by the business variables.

    The financial matrix is calculated as follows:
    - True Positive (TP): (Median Customer Value * Retention Rate) - Cost of Intervention
    - False Positive (FP): Cost of Intervention (Intervention on non-churner)
    - False Negative (FN): Median Customer Value (Lost value from missed churner)
    - True Negative (TN): 0 (Correctly ignored)

    Parameters
    ----------
    median_value (float): The median or average lifetime value (LTV) of a customer,
                          used as the value lost for a False Negative.
    cost (float): The cost of the intervention (e.g., the cost of the retention offer).
    retention_rate (float): The expected success rate of the intervention (e.g.,
                            if 40% of customers accept the offer and stay, use 0.40).
    labels (numpy.ndarray or pandas.Series): The true binary target labels (Y_test).
    predictions_proba (numpy.ndarray or pandas.Series): The predicted probabilities
                                                        for the positive class (Churn=1).

    Returns
    -------
    tuple
        - best_threshold (float): The prediction probability threshold (e.g., 0.650)
                                  that maximizes the total expected profit.
        - best_profit (float): The maximum total profit achieved at the best_threshold.
        - profit_values (dict): A dictionary mapping every threshold tested to its
                                corresponding total profit, useful for plotting the curve.
    """
    TP_value = (median_value * retention_rate) - cost
    FP_value = cost
    FN_value = median_value
    TN_value = 0

    if isinstance(predictions_proba, pd.Series):
        predictions_proba = predictions_proba.values

    if isinstance(labels, pd.Series):
        labels = labels.values

    safe_display(predictions_proba)

    thresholds = np.round(np.arange(0.050, 0.950 + 0.001, 0.001), 4)

    profit_values: Dict[float, float] = {}

    for t in thresholds:
        predictions = (predictions_proba > t).astype(int)

        matrix = confusion_matrix(labels, predictions, labels=[0, 1])

        if matrix.shape != (2, 2):
            continue

        TN = matrix[0, 0]
        FP = matrix[0, 1]
        FN = matrix[1, 0]
        TP = matrix[1, 1]

        profit = (TP * TP_value) - (FP * FP_value) - (FN * FN_value)

        profit_values[t] = profit

    if profit_values:
        best_threshold, best_profit = max(profit_values.items(), key=lambda item: item[1])

    else:
        best_threshold = 0.500
        best_profit = 0.0
        print("Warning: Could not calculate profit curve. Check input data.")


    profit_values = dict(sorted(profit_values.items(), key=lambda item: item[1], reverse=True))

    return best_threshold, best_profit, profit_values


def deployment_model(df: pd.DataFrame, model: Pipeline, target: str,
                     cols_to_drop: Optional[Union[str, List[str]]]) -> Pipeline:
    """
    Retrains the final, best-performing model on the entire dataset (X + y) and saves
    the complete, fitted scikit-learn Pipeline for deployment.

    This function executes the final step of the MLOps process by maximizing the
    information available to the model. It constructs a new Pipeline using the
    unfitted preprocessor and the tuned classifier, then fits this entire
    structure to all data (training and test combined). The final fitted Pipeline
    is saved to a .pkl file for production use.

    Parameters
    ----------
    df (pandas.DataFrame): The full, raw input DataFrame (contains X and y).
    model (sklearn.pipeline.Pipeline): The best-performing, fitted Pipeline object
                                       from the model comparison stage.
    target (str): The name of the target variable column.
    cols_to_drop (str or list, optional): Additional columns to drop from the
                                          features before retraining.

    Returns
    -------
    sklearn.pipeline.Pipeline: The final, fitted Pipeline object, ready for
                               making predictions on new, raw data.

    Side Effects
    ------------
    - Saves the fitted Pipeline object to a file named 'deployment_pipeline.pkl'
      using joblib.
    """

    if cols_to_drop is None:
        cols_to_drop_list: List[str] = []

    elif isinstance(cols_to_drop, str):
        cols_to_drop_list = [cols_to_drop]

    else:
        cols_to_drop_list = cols_to_drop

    try:
        X, y, _, _, _, _, preprocessor_unfitted, _ = preprocess_data(df, target=target,
                                                            cols_to_drop=cols_to_drop_list)

    except Exception as e:
        if 'preprocessor' in model.named_steps:
            preprocessor_unfitted = clone(model.named_steps['preprocessor'])
            X = df.drop(columns=[target] + cols_to_drop_list, errors='ignore')
            y = df[target]

        else:
            raise ValueError(f"Could not initialize data or extract preprocessor. Details {e}.")

    if 'classifier' not in model.named_steps:
        raise ValueError("Input model pipeline must contain a 'classifier' step.")


    classifier_unfitted = clone(model.named_steps['classifier'])

    deployment_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_unfitted),
        ('classifier', classifier_unfitted)
    ])

    print('Retraining Final Model on The Entire Dataset...')
    deployment_pipeline.fit(X, y)
    print('Retraining Complete.')

    try:
        joblib.dump(deployment_pipeline, 'deployment_pipeline.pkl')
        print("Deployment pipeline saved successfully to 'deployment_pipeline.pkl'.")

    except Exception as e:
        print("Error saving model with joblib: {e}.")

    return deployment_pipeline


def predict_churn(df: pd.DataFrame, threshold: float, model_path: str = 'deployment_pipeline.pkl') -> pd.DataFrame:
    """
    Loads the fitted deployment pipeline, predicts the churn probability for a
    dataset, flags customers for intervention, and filters the final target list.

    This function is designed to be the production step, taking raw customer data
    and returning a ranked list of high-risk customers who meet specific business
    criteria (Contract type).

    Parameters
    ----------
    df (pandas.DataFrame): The raw, unprocessed input customer data to score.
    threshold (float): The optimal churn probability threshold (e.g., 0.650)
                       determined by the Profit Curve analysis, above which
                       customers are flagged for intervention.
    model_path (str, optional): The file path to the saved, fitted deployment
                                pipeline (a scikit-learn Pipeline object saved
                                with joblib). Defaults to 'deployment_pipeline.pkl'.

    Returns
    -------
    pandas.DataFrame: A DataFrame containing only the customers who meet both
                      the model's high-risk criteria (Probability >= threshold)
                      and the business segment criteria (Contract == 'Month-to-month').
                      The DataFrame is sorted by 'Churn Probability' in descending
                      order, prioritizing the highest-risk customers.

    Side Effects
    ------------
    - Prints a message if the model file is not found.
    - Prints the count of customers identified for intervention.
    """

    loaded_model: Optional[Pipeline] = None

    try:
        loaded_model = joblib.load(model_path)

    except FileNotFoundError:
        print("Error: Model Path Not Found")
        return pd.DataFrame()

    except Exception as e:
        print(f"Error loading model: {e}.")
        return pd.DataFrame()

    try:
        proba = loaded_model.predict_proba(df)[:,1]

    except Exception as e:
        print(f"Error during prediction: {e}.")
        return pd.DataFrame()

    df_scored = df.copy()
    df_scored['Churn Probability'] = proba
    df_scored['Intervention Flag'] = (proba >= threshold).astype(int)

    test_target = df_scored[(df_scored['Intervention Flag'] == 1) & (df_scored['Contract'] == 'Month-to-month')].copy()

    print(f'Identified {len(test_target)} customers for A/B/C intervention test.')

    return test_target.sort_values(by='Churn Probability', ascending=False)


def abc_test(high_risk_target: pd.DataFrame, random_seed: int = 123) -> pd.DataFrame:
    """
    Randomly assigns a pre-filtered population of high-risk customers into
    three groups (A/Control, B/Price Offer, C/Service Offer) for an intervention test.

    This function is typically run after a predictive model has identified and
    filtered the final, actionable customer segment (e.g., customers with a
    high probability of churn AND a 'Month-to-month' contract). The assignment
    is balanced and randomized using the provided seed.

    Parameters
    ----------
    high_risk_target : pandas.DataFrame
        The DataFrame containing the final, pre-filtered set of customers to be
        targeted for the A/B/C test. An empty DataFrame will result in an
        early return with a message.

    random_seed : int, default=123
        Seed for the NumPy random number generator to ensure reproducible
        assignment of customers to test groups.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame augmented with two new columns:
        - **'Group'**: The randomized test group assignment, e.g., 'A (Control)'.
        - **'Intervention Details'**: A specific description of the offer
          associated with the group ('10% Off 1-Year Contract', '6 Months Free Tech Support',
          or 'None (Control)').

    Notes
    -----
    - The randomization uses ``numpy.random.choice`` with replacement, but since
      the sample size is the total population size, this effectively performs a
      simple random sample without replacement *if* all groups are present.
      The sample is intended to be a simple random assignment into the three
      categories (A, B, C).
    - A summary table of the assignment count per group is printed to the console.
    """

    if high_risk_target.empty:
        print('Dataframe empty. No customers assign to test groups.')
        return pd.DataFrame()

    target_population = high_risk_target.copy()
    n = len(target_population)

    if n == 0:
        print('No high-risk customers found in the target segment for A/B/C testing.')
        return pd.DataFrame()

    groups = ['A (Control)', 'B (Price Offer)', 'C (Service Offer)']

    np.random.seed(random_seed)
    group_assignment = np.random.choice(groups, size=n, replace=True)

    target_population['Group'] = group_assignment

    def map_intervention(group):
        if group == 'B (Price Offer)':
            return '10% Off 1-Year Contract'

        elif group == 'C (Service Offer)':
            return '6 Months Free Tech Support'

        else:
            return 'None (Control)'

    target_population['Intervention Details'] = target_population['Group'].apply(map_intervention)

    summary = target_population['Group'].value_counts().reset_index()
    summary.columns = ['Group', 'Customer Count']
    print('---Test Group Assignment Summary')
    print(summary.to_markdown(index=False))
    return target_population

