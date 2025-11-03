import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import shap
import io
import base64

try:
  from xgboost import XGBClassifier
except ImportError:
  class XGBClassifier: pass

try:
  from lightgbm import LGBMClassifier
except ImportError:
  class LGBMClassifier: pass


def model_global_explainer(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           n_samples: int = 500, n_clusters: int = 5, max_display: int = 10,
                           random_state: int  = 123) -> None:
    """
    Generates and plots **global feature importance** using SHAP (SHapley Additive exPlanations)
    values for the final classifier within a fitted scikit-learn Pipeline.

    This function isolates the fitted classifier and explains its predictions
    on the preprocessed feature data. It calculates the SHAP values for the entire
    test set to determine **global feature importance** (i.e., feature impact across all
    predictions). It dynamically selects between ``shap.TreeExplainer`` (for tree-based models)
    and ``shap.KernelExplainer`` (for others, using a k-means sampled background dataset
    to manage computational load).

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fitted scikit-learn Pipeline object containing the 'preprocessor' and
        'classifier' named steps.
    X_train : pandas.DataFrame
        The raw training feature set (unprocessed data). Used to define the feature
        names and the background dataset for KernelExplainer.
    X_test : pandas.DataFrame
        The raw testing feature set (unprocessed data). Used as the primary data
        to calculate global SHAP values.
    n_samples : int, optional
        The number of samples to select from X_train for the explainer's background
        dataset when using KernelExplainer. Defaults to 500.
    n_clusters : int, optional
        The number of clusters to use for k-means sampling of the background dataset.
        Defaults to 5.
    max_display : int, optional
        The maximum number of features to display in the final SHAP summary plot.
        Defaults to 10.
    random_state : int, optional
        The seed used by the random number generator for sampling and k-means
        clustering (if used). Defaults to 123.

    Returns
    -------
    None
        The function directly generates and displays a **SHAP Summary Plot**
        (global feature importance as a bar chart) using ``shap.summary_plot``.

    Raises
    ------
    ValueError
        If the `model` is not an ``sklearn.pipeline.Pipeline`` or does not
        contain both 'preprocessor' and 'classifier' named steps.
    AttributeError
        If the 'preprocessor' step lacks a reliable feature name retrieval method
        (e.g., `get_feature_names_out`).
    """

    if not isinstance(model, Pipeline):
        raise ValueError(f"ValueError: The 'model' must be a sklearn.pipeline.Pipeline object, but received {type(model).__name__}.")

    if 'classifier' not in model.named_steps or 'preprocessor' not in model.named_steps:
        raise ValueError(f"ValueError: Pipeline must contain steps named 'preprocessor' and 'classifier'.")

    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    try:
        feature_names = preprocessor.get_feature_names_out()

    except (AttributeError, TypeError) as e:
        raise AttributeError(f"AttributeError: Failed to retrieve feature names. Ensure 'preprocessor' has 'get_feature_names_out()'. Details: {e}.")

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    X_test_for_shap: pd.DataFrame = X_test_transformed_df

    tree_models = (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier)

    explainer_type: str

    if isinstance(classifier, tree_models):
            explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(X_test_for_shap)
            explainer_type = 'TreeExplainer'
    else:
        sample_size = min(n_samples, len(X_train_transformed_df))
        X_train_sample = X_train_transformed_df.sample(n=sample_size, random_state=random_state)
        X_background = shap.kmeans(X_train_sample.values, n_clusters).data

        if len(X_test_transformed_df) > 1000:
            print("Warning: Limiting KernelExplainer test set calculation to 1000 samples for performance.")
            X_test_for_shap = X_test_transformed_df.sample(n=1000, random_state=random_state)

        model_predict_proba = classifier.predict_proba
        explainer = shap.KernelExplainer(model_predict_proba, X_background)
        shap_values = explainer.shap_values(X_test_for_shap)
        explainer_type = 'KernelExplainer'

    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_to_plot = shap_values[1]
        plot_title = f'Global SHAP Feature Importance (Positive Class) - Explainer: {explainer_type}'

    else:
        shap_to_plot = shap_values
        plot_title = f'Global SHAP Feature Importance - Explainer: {explainer_type}'

    fig, ax = plt.subplots(figsize=(12, 8))

    shap.summary_plot(shap_to_plot, X_test_for_shap, show=False, max_display=max_display, plot_type='bar')

    plt.title(plot_title)
    plt.xlabel('Mean Absolute SHAP Value (Global Impact on Model Output)')

    plt.tight_layout()
    plt.show()


def model_global_explainer_app(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               n_samples: int = 500, n_clusters: int = 5, max_display: int = 10,
                               random_state: int  = 123) -> None:
    """
    Generates and visualizes **global feature importance** using SHAP (SHapley Additive exPlanations)
    for the classifier within a fitted scikit-learn Pipeline.

    This function extracts the classifier from a fitted Pipeline and explains its predictions
    on preprocessed feature data. It computes SHAP values for the test set to determine 
    **global feature importance** (i.e., the average impact of each feature across all predictions). 
    The function dynamically selects between ``shap.TreeExplainer`` (for tree-based models)
    and ``shap.KernelExplainer`` (for other models, using k-means sampled background data
    to reduce computational cost).

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A fitted scikit-learn Pipeline object containing the named steps 'preprocessor' and 'classifier'.
    X_train : pandas.DataFrame
        The raw training feature set (unprocessed). Used to obtain feature names and to construct
        the background dataset for KernelExplainer.
    X_test : pandas.DataFrame
        The raw testing feature set (unprocessed). Used to compute global SHAP values.
    n_samples : int, optional
        Number of samples to draw from X_train for KernelExplainer's background dataset. Default is 500.
    n_clusters : int, optional
        Number of clusters for k-means sampling of the background dataset. Default is 5.
    max_display : int, optional
        Maximum number of features to display in the SHAP summary plot. Default is 10.
    random_state : int, optional
        Random seed for sampling and k-means clustering. Default is 123.

    Returns
    -------
    str
        A base64-encoded PNG image of the SHAP summary plot showing global feature importance.

    Raises
    ------
    ValueError
        If `model` is not a scikit-learn Pipeline or does not contain both 'preprocessor' 
        and 'classifier' steps.
    AttributeError
        If the 'preprocessor' step does not provide `get_feature_names_out()`.
    """

    if not isinstance(model, Pipeline):
        raise ValueError(f"ValueError: The 'model' must be a sklearn.pipeline.Pipeline object, but received {type(model).__name__}.")

    if 'classifier' not in model.named_steps or 'preprocessor' not in model.named_steps:
        raise ValueError(f"ValueError: Pipeline must contain steps named 'preprocessor' and 'classifier'.")

    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    try:
        feature_names = preprocessor.get_feature_names_out()

    except (AttributeError, TypeError) as e:
        raise AttributeError(f"AttributeError: Failed to retrieve feature names. Ensure 'preprocessor' has 'get_feature_names_out()'. Details: {e}.")

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    X_test_for_shap: pd.DataFrame = X_test_transformed_df

    tree_models = (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier)

    explainer_type: str

    if isinstance(classifier, tree_models):
            explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(X_test_for_shap)
            explainer_type = 'TreeExplainer'
    else:
        sample_size = min(n_samples, len(X_train_transformed_df))
        X_train_sample = X_train_transformed_df.sample(n=sample_size, random_state=random_state)
        X_background = shap.kmeans(X_train_sample.values, n_clusters).data

        if len(X_test_transformed_df) > 1000:
            print("Warning: Limiting KernelExplainer test set calculation to 1000 samples for performance.")
            X_test_for_shap = X_test_transformed_df.sample(n=1000, random_state=random_state)

        model_predict_proba = classifier.predict_proba
        explainer = shap.KernelExplainer(model_predict_proba, X_background)
        shap_values = explainer.shap_values(X_test_for_shap)
        explainer_type = 'KernelExplainer'

    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_to_plot = shap_values[1]
        plot_title = f'Global SHAP Feature Importance (Positive Class) - Explainer: {explainer_type}'

    else:
        shap_to_plot = shap_values
        plot_title = f'Global SHAP Feature Importance - Explainer: {explainer_type}'

    fig, ax = plt.subplots(figsize=(12, 8))

    shap.summary_plot(shap_to_plot, X_test_for_shap, show=False, max_display=max_display, plot_type='bar')

    plt.title(plot_title)
    plt.xlabel('Mean Absolute SHAP Value (Global Impact on Model Output)')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"


def model_local_explainer(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, index: int = 0,
                          n_samples: int = 500, n_clusters: int = 5, max_display: int = 10,
                          random_state: int = 123) -> None:
    """
    Generates and plots **local feature contributions** using SHAP (SHapley Additive exPlanations)
    values for a single prediction instance from the final classifier within a fitted scikit-learn Pipeline.

    This function isolates the fitted classifier and preprocessor to explain a single prediction.
    It calculates the SHAP values for the instance specified by **'index'** in the test set to
    determine the **local feature contribution** (i.e., how each feature shifts the prediction
    from the average expected value). It dynamically selects between ``shap.TreeExplainer``
    and ``shap.KernelExplainer`` (using k-means sampling for efficiency).

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fitted scikit-learn Pipeline object containing the 'preprocessor' and
        'classifier' named steps.
    X_train : pandas.DataFrame
        The raw training feature set (unprocessed data). Used for the background
        dataset sampling.
    X_test : pandas.DataFrame
        The raw testing feature set (unprocessed data). Used to select the single instance
        for local explanation.
    index : int, optional
        The **row index** (from the raw X_test DataFrame) of the single prediction instance
        to be explained. Defaults to 0.
    n_samples : int, optional
        The number of samples selected from X_train for the explainer's background
        dataset when using KernelExplainer. Defaults to 500.
    n_clusters : int, optional
        The number of clusters to use for k-means sampling of the background dataset.
        Defaults to 5.
    max_display : int, optional
        The maximum number of features to display in the SHAP waterfall plot.
        Defaults to 10.
    random_state : int, optional
        The seed used for sampling and k-means clustering. Defaults to 123.

    Returns
    -------
    None
        The function directly generates and displays a **SHAP Waterfall Plot** showing the
        contribution of features to the specific prediction for the instance at the given index.

    Raises
    ------
    ValueError
        If the `model` object is not a valid Pipeline, is missing steps, or if the
        `index` is out of bounds for the test data.
    AttributeError
        If the 'preprocessor' step fails to retrieve feature names (e.g., lacks
        `get_feature_names_out`).
    """

    if not isinstance(model, Pipeline):
        raise ValueError(f"ValueError: The 'model' must be a sklearn.pipeline.Pipeline object, but received {type(model).__name__}.")

    if 'classifier' not in model.named_steps or 'preprocessor' not in model.named_steps:
        raise ValueError(f"ValueError: Pipeline must contain steps named 'preprocessor' and 'classifier'.")

    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    try:
        feature_names = preprocessor.get_feature_names_out()

    except (AttributeError, TypeError) as e:
        raise AttributeError(f"AttributeError: Failed to retrieve feature names. Ensure 'preprocessor' has 'get_feature_names_out()'. Details: {e}.")

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    tree_models = (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier)

    if not (0 <= index < len(X_test_transformed_df)):
        raise ValueError(f"ValueError: Index {index} is out of bounds. Must be between 0 and {len(X_test_transformed_df) - 1}.")

    test_instance = X_test_transformed_df.iloc[[index]]
    customer_index_label = X_test_transformed_df.index[index]

    explainer_type: str

    if isinstance(classifier, tree_models):
        explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent')
        shap_values = explainer.shap_values(test_instance)
        expected_value = explainer.expected_value
        explainer_type = 'TreeExplainer'

    else:
        sample_size = min(n_samples, len(X_train_transformed_df))
        X_train_sample = X_train_transformed_df.sample(n=n_samples, random_state=random_state)
        X_background = shap.kmeans(X_train_sample, n_clusters).data
        model_predict_proba = classifier.predict_proba
        explainer = shap.KernelExplainer(model_predict_proba, X_background)
        shap_values = explainer.shap_values(test_instance, silent=True)
        expected_value = explainer.expected_value
        explainer_type = 'KernelExplainer'

    if isinstance(shap_values, list) and len(shap_values) > 1:
        e_values = expected_value[1]
        s_values = shap_values[1][0]
        plot_title = f"Local SHAP Contribution for Positive Class Prediction"

    else:
        e_values = expected_value
        s_values = shap_values[0]
        plot_title = f"Local SHAP Contribution for Prediction"

    shap_explanation = shap.Explanation(values=s_values, base_values=e_values, data=test_instance.values[0],
                                        feature_names=feature_names)

    fig_height = max(8, max_display * 0.7)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    shap.waterfall_plot(shap_explanation, max_display=max_display, show=False)

    plt.title(f"{plot_title} (Instance Index: {customer_index_label}) - Explainer: {explainer_type}")
    plt.tight_layout()
    plt.show()


def model_local_explainer_app(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, index: int = 0,
                          n_samples: int = 500, n_clusters: int = 5, max_display: int = 10,
                          random_state: int = 123) -> None:
    """
    Generates and visualizes **local feature contributions** using SHAP (SHapley Additive exPlanations)
    for a single prediction instance from a fitted scikit-learn Pipeline.

    This function extracts the classifier and preprocessor from the Pipeline to explain a single prediction.
    It computes SHAP values for the instance specified by **'index'** in the test set, showing the 
    **local feature contribution** (i.e., how each feature shifts the prediction from the expected value). 
    The function dynamically selects between ``shap.TreeExplainer`` (for tree-based models) and 
    ``shap.KernelExplainer`` (for other models, using k-means sampled background data for efficiency).

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A fitted scikit-learn Pipeline containing the named steps 'preprocessor' and 'classifier'.
    X_train : pandas.DataFrame
        Raw training feature set (unprocessed). Used for background dataset sampling in KernelExplainer.
    X_test : pandas.DataFrame
        Raw testing feature set (unprocessed). The instance at `index` will be explained.
    index : int, optional
        Row index of the single prediction instance to explain. Default is 0.
    n_samples : int, optional
        Number of samples to select from X_train for KernelExplainer's background dataset. Default is 500.
    n_clusters : int, optional
        Number of clusters for k-means sampling of the background dataset. Default is 5.
    max_display : int, optional
        Maximum number of features to display in the SHAP waterfall plot. Default is 10.
    random_state : int, optional
        Random seed for sampling and k-means clustering. Default is 123.

    Returns
    -------
    str
        A base64-encoded PNG image of the SHAP waterfall plot showing local feature contributions
        for the specified instance.

    Raises
    ------
    ValueError
        If `model` is not a scikit-learn Pipeline, lacks required steps, or if `index` is out of bounds.
    AttributeError
        If the 'preprocessor' step cannot provide feature names (e.g., missing `get_feature_names_out()`).
    """

    if not isinstance(model, Pipeline):
        raise ValueError(f"ValueError: The 'model' must be a sklearn.pipeline.Pipeline object, but received {type(model).__name__}.")

    if 'classifier' not in model.named_steps or 'preprocessor' not in model.named_steps:
        raise ValueError(f"ValueError: Pipeline must contain steps named 'preprocessor' and 'classifier'.")

    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    try:
        feature_names = preprocessor.get_feature_names_out()

    except (AttributeError, TypeError) as e:
        raise AttributeError(f"AttributeError: Failed to retrieve feature names. Ensure 'preprocessor' has 'get_feature_names_out()'. Details: {e}.")

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    tree_models = (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier)

    if not (0 <= index < len(X_test_transformed_df)):
        raise ValueError(f"ValueError: Index {index} is out of bounds. Must be between 0 and {len(X_test_transformed_df) - 1}.")

    test_instance = X_test_transformed_df.iloc[[index]]
    customer_index_label = X_test_transformed_df.index[index]

    explainer_type: str

    if isinstance(classifier, tree_models):
        explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent')
        shap_values = explainer.shap_values(test_instance)
        expected_value = explainer.expected_value
        explainer_type = 'TreeExplainer'

    else:
        sample_size = min(n_samples, len(X_train_transformed_df))
        X_train_sample = X_train_transformed_df.sample(n=n_samples, random_state=random_state)
        X_background = shap.kmeans(X_train_sample, n_clusters).data
        model_predict_proba = classifier.predict_proba
        explainer = shap.KernelExplainer(model_predict_proba, X_background)
        shap_values = explainer.shap_values(test_instance, silent=True)
        expected_value = explainer.expected_value
        explainer_type = 'KernelExplainer'

    if isinstance(shap_values, list) and len(shap_values) > 1:
        e_values = expected_value[1]
        s_values = shap_values[1][0]
        plot_title = f"Local SHAP Contribution for Positive Class Prediction"

    else:
        e_values = expected_value
        s_values = shap_values[0]
        plot_title = f"Local SHAP Contribution for Prediction"

    shap_explanation = shap.Explanation(values=s_values, base_values=e_values, data=test_instance.values[0],
                                        feature_names=feature_names)

    fig_height = max(8, max_display * 0.7)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    shap.waterfall_plot(shap_explanation, max_display=max_display, show=False)

    plt.title(f"{plot_title} (Instance Index: {customer_index_label}) - Explainer: {explainer_type}")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"
