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
import os

from .utils import (safe_display)

def preprocess_data(df: pd.DataFrame, target: str,
                    cols_to_drop: Optional[Union[str, List[str]]] = None, test_size: float = 0.2,
                    random_state: int = 123) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, ColumnTransformer]:
    """
    Prepares data for machine learning by splitting and defining the preprocessing logic.

    This function performs key data preparation steps, including splitting, feature
    selection based on cardinality and type, and defining a ColumnTransformer
    for preprocessing.

    Preprocessing Logic:
    - One-Hot Encoding ('one_hot'): Applied to categorical/object columns with 5 or fewer unique values, including 'City'.
    - Standard Scaling ('scaler'): Applied to numerical columns (int64/float64) with more than 5 unique values.
    - Remainder ('passthrough'): All other columns (including high-cardinality categoricals and low-cardinality numerics) are passed through unchanged.

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
    tuple
        - X (pandas.DataFrame): The full features DataFrame.
        - y (pandas.Series): The full target Series.
        - X_train (pandas.DataFrame): The training features.
        - X_test (pandas.DataFrame): The testing features.
        - y_train (pandas.Series): The training target.
        - y_test (pandas.Series): The testing target.
        - preprocessor (ColumnTransformer): The **unfitted** ColumnTransformer object.
        - preprocessor_trained (ColumnTransformer): The **fitted** ColumnTransformer object (fitted only on X_train).

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

        one_hot_cols = [col for col in X.columns if (X[col].nunique() <= 5 and X[col].dtype in ['object', 'category', 'bool']) or col == 'City']

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

        scaler_cols = [col for col in numeric_cols if X[col].nunique() > 5]


        preprocessor = ColumnTransformer(
            transformers=[
                ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_cols),
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
    preprocessor : ColumnTransformer
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
    Pipelines without a predefined hyperparameter grid are skipped with a warning.

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
        hyperparameter grid ('params').

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
    Train and evaluate multiple machine learning models with hyperparameter tuning.

    Iterates through a dictionary of models and their hyperparameter grids, performing
    GridSearchCV for each. Trains the best estimator, evaluates predictions on the
    test set, computes standard classification metrics (Recall, Precision, F1, Accuracy,
    ROC-AUC), and identifies the top 4 most important features where possible.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Test target.
    params : dict
        Dictionary with model pipelines and hyperparameter grids. Each entry should have
        keys `'model'` (Pipeline) and `'params'` (dict of hyperparameters).
    metric : str, default 'f1'
        Scoring metric used for GridSearchCV.
    cv : int, default 5
        Number of cross-validation folds.

    Returns
    -------
    best_models : dict
        Dictionary of the best trained estimators, keyed by model name.
    model_results : list
        List of dictionaries containing evaluation metrics, best parameters, top features,
        predictions, and probabilities for each model.

    Raises
    ------
    TypeError
        If input data types are incorrect.
    ValueError
        If the `params` dictionary is empty or CV is invalid.
    RuntimeError
        For unexpected errors during training or evaluation.
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
    Train and evaluate a soft VotingClassifier ensemble from base models.

    Constructs a VotingClassifier with the provided best models, performs RandomizedSearchCV
    over possible weight combinations to find optimal base model weights, and evaluates
    performance on the test set using standard metrics (Accuracy, Recall, Precision, F1, ROC-AUC).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Test target.
    best_models : dict
        Dictionary of trained base models (Pipeline objects) to include in the ensemble.
    metric : str, default 'f1'
        Scoring metric used for RandomizedSearchCV.
    n_iter : int, default 10
        Number of weight combinations sampled in RandomizedSearchCV.
    cv : int, default 5
        Number of cross-validation folds.
    random_state : int, default 123
        Random seed for reproducibility.

    Returns
    -------
    best_voting_model : dict
        Dictionary containing the trained VotingClassifier under the key 'VotingClassifier'.
    vot_model_results : list
        List with a single dictionary containing evaluation metrics, best parameters,
        predictions, and prediction probabilities for the VotingClassifier.

    Notes
    -----
    If one or more base models do not support `predict_proba`, the VotingClassifier
    may fall back to 'hard' voting automatically.
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
    Orchestrate training, evaluation, and comparison of multiple classification models.

    Performs a complete workflow including:
    - Data preprocessing
    - Baseline Logistic Regression cross-validation
    - Hyperparameter tuning of multiple comparative models
    - Model evaluation and ranking
    - Construction of a VotingClassifier ensemble
    - Identification of the best model based on a specified metric

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str
        Name of the target variable column.
    comparative_models : dict
        Dictionary of scikit-learn classifier objects to compare against Logistic Regression.
    cols_to_drop : str or list, optional
        Columns to exclude from features. Defaults to None.
    metric : str, default 'f1'
        Metric for evaluating models and sorting results.
    test_size : float, default 0.2
        Fraction of data reserved for testing.
    random_state : int, default 123
        Seed for reproducibility.

    Returns
    -------
    all_models : dict
        Dictionary of best trained models, including the final VotingClassifier.
    all_results : pd.DataFrame
        DataFrame summarizing metrics, best parameters, and top features for all models.
    model_object_by_metric : Pipeline
        Best-performing model based on the specified metric.
    X_train : pd.DataFrame
        Training features used.
    X_test : pd.DataFrame
        Testing features used.
    y_test : pd.Series
        Test target variable.

    Raises
    ------
    ValueError, TypeError, RuntimeError
        If issues occur during preprocessing, training, evaluation, or saving models.

    Notes
    -----
    This function relies on external helper functions: `preprocess_data`, `model_pipelines`,
    `hyperparameters`, `train_evaluate_model`, and `voting_model`.
    """

    all_models = {}
    all_results = pd.DataFrame()
    model_object_by_metric: Union[Pipeline, None] = None
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_test = pd.Series(dtype='int')
    best_model_name = 'N/A'

    if os.path.exists('model_results.pkl'):
        try:
            print("Loading model... (skipping training)")
            saved_data = joblib.load('model_results.pkl')

            all_models = saved_data['all_models']
            all_results = saved_data['all_results']
            model_object_by_metric = saved_data['model_untrained']
            X_train = saved_data['X_train']
            X_test = saved_data['X_test']
            y_test = saved_data['y_test']

            print('\n\n--- Model Results ---')
            print(all_results[['name', 'recall', 'precision', 'accuracy', 'f1', 'roc_auc']])
            print(f"\n\n--- Best Model: {all_results['name'].iloc[0]} ---\n\n")
            
            return all_models, all_results, model_object_by_metric, X_train, X_test, y_test
            
        except (EOFError, FileNotFoundError, KeyError, joblib.externals.loky.process_executor.TerminatedWorkerError) as e:
            print(f'Saved model file is corrupted or incomplete: {type(e).__name__} - {e}. Retraining model...')

        except Exception as e:
            print(f'Unexpected error loading saved model: {type(e).__name__} - {e}. Retraining model...')

    try:
        print('Start training models ...')
        X, y, X_train, X_test, y_train, y_test, preprocessor, preprocessor_trained = preprocess_data(df, target, cols_to_drop, test_size, random_state)

        models = model_pipelines(preprocessor, comparative_models, random_state)
        params = hyperparameters(models)

        baseline_score = cross_val_score(models['LogisticRegression'], X, y, cv=5, scoring=metric)
        print('-' * 100)
        print(f'\nBaseline Logistic Regression Score: {np.mean(baseline_score):.4f}.\n')
        print('-' * 100)

        print('\n','-' * 50,'Comparative Models Results','-' * 50,'\n')
        print('Dtypes,', X_train.dtypes, X_train.dtypes, y_test.dtypes)
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

        try:
            joblib.dump({
                'all_models': all_models,
                'all_results': all_results,
                'model_untrained': model_object_by_metric,
                'X_train': X_train,
                'X_test': X_test,
                'y_test': y_test
                },
                'model_results.pkl'
                )
            print("Deployment pipeline saved successfully to 'model_results.pkl'.")

        except Exception as e:
            print("Error saving model with joblib: {e}.")

        return all_models, all_results, model_object_by_metric, X_train, X_test, y_test

    except NotImplementedError as e:
        print(f"Execution failed: A required helper function is missing: {e}.")
        all_models = {}
        all_results = pd.DataFrame()
        model_object_by_metric = None
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_test = pd.Series(dtype='int')

        return all_models, all_results, model_object_by_metric, X_train, X_test, y_test

    except Exception as e:
        print(f"An unexpected error occurred during the workflow: {type(e).__name__} - {e}.")
        all_models = {}
        all_results = pd.DataFrame()
        model_object_by_metric = None
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_test = pd.Series(dtype='int')
        
        return all_models, all_results, model_object_by_metric, X_train, X_test, y_test
    

## MLOPS, Optimization and Business Action Functions


def profit_curve(model_df: pd.DataFrame, median_value: float, cost: float, retention_rate: float,
                 labels: Union[np.ndarray, pd.Series]):
    """
    Computes the optimal probability threshold and maximum expected profit
    for a classification model using a cost-sensitive profit evaluation.

    Assumes the predicted probabilities for the positive class are stored
    in the 'predictions_proba' column of the first row of the input DataFrame.

    Profit calculation logic:
    - True Positive (TP): (median_value * retention_rate) - cost
    - False Positive (FP): - cost
    - False Negative (FN): - median_value
    - True Negative (TN): 0

    Parameters
    ----------
    model_df : pandas.DataFrame
        DataFrame with a 'predictions_proba' column containing predicted probabilities
        for the positive class in its first row.
    median_value : float
        Median customer lifetime value (CLTV) for estimating opportunity cost.
    cost : float
        Cost of an intervention (e.g., retention offer).
    retention_rate : float
        Probability that an intervention successfully prevents churn.
    labels : array-like
        Ground truth target labels (0 or 1).

    Returns
    -------
    best_threshold : float
        Probability threshold that maximizes expected profit.
    best_profit : float
        Maximum achievable profit at the optimal threshold.
    profit_values : dict
        Mapping of thresholds to calculated profits, sorted descending.
    """

    if not isinstance(model_df, pd.DataFrame):
        raise TypeError("'model_df' must be a pandas DataFrame.")
    
    if 'predictions_proba' not in model_df.columns:
        raise ValueError("'model_df' must contain a 'predictions_proba' column.")
    
    if not isinstance(median_value, (int, float)) or median_value < 0:
        raise ValueError("'median_value' must be a non-negative number.")
    
    if not isinstance(cost, (int, float)) or cost < 0:
        raise ValueError("'cost' must be a non-negative number.")
    
    if not (0 <= retention_rate <= 1):
        raise ValueError("'retention_rate' must be a float between 0 and 1.")
    
    predictions_proba = model_df.iloc[0]['predictions_proba']

    if isinstance(predictions_proba, pd.Series):
        predictions_proba = predictions_proba.values
    
    elif not isinstance(predictions_proba, np.ndarray):
        predictions_proba = np.array(predictions_proba)

    if predictions_proba.ndim != 1:
        raise ValueError("Expected 1D 'predictions_proba' array.")

    if isinstance(labels, pd.Series):
        labels = labels.values
    
    TP_value = (median_value * retention_rate) - cost
    FP_value = cost
    FN_value = median_value
    TN_value = 0

    thresholds = np.round(np.arange(0.050, 0.950 + 0.001, 0.001), 4)
    profit_values: Dict[float, float] = {}

    for t in thresholds:
        predictions = (predictions_proba > t).astype(int)

        matrix = confusion_matrix(labels, predictions, labels=[0, 1])

        if matrix.shape != (2, 2):
            continue

        TN, FP = matrix[0]
        FN, TP = matrix[1]

        profit = (TP * TP_value) - (FP * FP_value) - (FN * FN_value)

        profit_values[t] = profit

    if profit_values:
        best_threshold, best_profit = max(profit_values.items(), key=lambda item: item[1])

    else:
        best_threshold, best_profit = 0.500, 0.0
        print("Warning: Could not calculate profit curve. Check input data.")


    profit_values = dict(sorted(profit_values.items(), key=lambda item: item[1], reverse=True))

    return best_threshold, best_profit, profit_values


def deployment_model(df: pd.DataFrame, model: Pipeline, target: str,
                     cols_to_drop: Optional[Union[str, List[str]]]) -> Pipeline:
    """
    Retrains the final, best-performing model on the entire dataset (X + y)
    and saves the fully fitted scikit-learn Pipeline for deployment.

    Constructs a new Pipeline with the unfitted preprocessor and classifier
    and fits it to all available data. Saves the trained Pipeline to
    'deployment_pipeline.pkl'.

    Parameters
    ----------
    df : pandas.DataFrame
        Full raw dataset containing features and target.
    model : sklearn.pipeline.Pipeline
        Best-performing, fitted Pipeline from model comparison stage.
    target : str
        Name of the target variable column.
    cols_to_drop : str or list, optional
        Columns to drop from features before retraining.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fully fitted Pipeline ready for deployment.

    Side Effects
    ------------
    Saves the fitted Pipeline to 'deployment_pipeline.pkl'.
    """

    if os.path.exists("deployment_pipeline.pkl"):
        print("Deployment model already exists - loading existing pipeline (skipping retraining and deployment).")
        try:
            existing_pipeline = joblib.load("deployment_pipeline.pkl")

            return existing_pipeline
        
        except (EOFError, FileNotFoundError, KeyError, joblib.externals.loky.process_executor.TerminatedWorkerError) as e:
            print(f'Warning: Could not load existing deployment pipeline ({type(e).__name__}), retraining and deploying.')
        
        except Exception as e:
            print(f"Unexpected error loading deployment pipeline ({type(e).__name__}), retraining and deploying.")
    
    if cols_to_drop is None:
        cols_to_drop_list: List[str] = []

    elif isinstance(cols_to_drop, str):
        cols_to_drop_list = [cols_to_drop]

    else:
        cols_to_drop_list = cols_to_drop

    try:
        X, y, X_train, X_test, _, y_test, preprocessor_unfitted, _ = preprocess_data(df, target=target,
                                                            cols_to_drop=cols_to_drop_list)

    except Exception as e:
        if 'preprocessor' in model.named_steps:
            preprocessor_unfitted = clone(model.named_steps['preprocessor'])
            X = df.drop(columns=[target] + cols_to_drop_list, errors='ignore')
            y = df[target]

            preprocessor_unfitted.fit(X)
            
            X_train, X_test, y_test = None, None, None

        else:
            raise ValueError(f"Could not initialize data or extract preprocessor. Details {e}.")

    if 'classifier' not in model.named_steps:
        raise ValueError("Input model pipeline must contain a 'classifier' step.")


    classifier_unfitted = clone(model.named_steps['classifier'])

    deployment_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_unfitted),
        ('classifier', classifier_unfitted)
    ])

    print('Start retraining of final model on full dataset...')
    deployment_pipeline.fit(X, y)
    print('Retraining complete.')

    try:
        joblib.dump(deployment_pipeline,'deployment_pipeline.pkl')
        print("Deployment pipeline saved successfully to 'deployment_pipeline.pkl'.")

    except Exception as e:
        print("Error saving model with joblib: {e}.")

    return deployment_pipeline


def predict_churn(df: pd.DataFrame, threshold: float, model_path: str = 'deployment_pipeline.pkl') -> pd.DataFrame:
    """
    Predicts churn probabilities using a deployed model, flags high-risk customers,
    and returns a ranked list for intervention.

    Loads a fitted deployment pipeline, calculates probabilities, applies
    the optimal threshold, and adds an intervention flag.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw customer data to score.
    threshold : float
        Probability threshold for flagging customers (from Profit Curve).
    model_path : str, optional
        Path to the saved deployment pipeline. Defaults to 'deployment_pipeline.pkl'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all customers with predicted churn probability and
        intervention flag. Sorted by 'Churn Probability' descending.

    Side Effects
    ------------
    Prints messages if the model cannot be loaded or if prediction fails.
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

    threshold = float(threshold)
    df_scored = df.copy()
    df_scored['Churn Probability'] = proba
    df_scored['Intervention Flag'] = (proba >= threshold).astype(int)

    return df_scored.sort_values(by='Churn Probability', ascending=False)


def abc_test(predicted_df: pd.DataFrame, random_seed: int = 123) -> pd.DataFrame:
    """
    Randomly assigns a pre-filtered population of high-risk customers into
    three groups (A/Control, B/Price Offer, C/Service Offer) for an intervention test.

    This function is typically run after a predictive model has identified and
    filtered the actionable customer segment (e.g., customers with high
    probability of churn AND a 'Month-to-month' contract). The assignment
    is balanced and randomized using the provided seed.

    Parameters
    ----------
    predicted_df : pandas.DataFrame
        DataFrame containing the final, pre-filtered set of customers to be
        targeted for the A/B/C test. An empty DataFrame will result in an
        early return with a message.
    random_seed : int, default=123
        Seed for the NumPy random number generator to ensure reproducible
        assignment of customers to test groups.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame augmented with two new columns:
        - **'Group'**: Randomized test group assignment (e.g., 'A (Control)').
        - **'Intervention Details'**: Description of the offer associated with
          the group ('10% Off 1-Year Contract', '6 Months Free Tech Support',
          or 'None (Control)').

    Notes
    -----
    - Randomization uses ``numpy.random.choice`` with replacement, but since
      the sample size equals the total population, it effectively performs
      a simple random assignment without replacement.
    - A summary table of the assignment count per group is printed to the console.
    """

    if predicted_df.empty:
        print('Dataframe empty. No customers assign to test groups.')
        return pd.DataFrame()

    target_population = predicted_df[(predicted_df['Intervention Flag'] == 1) & (predicted_df['Contract'] == 'Month-to-month')].copy()

    print(f'Identified {len(target_population)} customers for A/B/C intervention test.')
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

