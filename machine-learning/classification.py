from warnings import warn
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_validate


plt.style.use('seaborn-whitegrid')


def fit_eval_clf_with_CV(estimator, X: pd.DataFrame, y: pd.Series, report_coefs=True) -> Tuple[dict, pd.Series]:
    """ Fit and evaluate a classifier model using CV to obtain realistic estimates of performance.

    Returns:
        scores (dict):
        preds (pd.Series): Predicted probability of a positive classification.

    """

    y_proba = pd.Series(cross_val_predict(estimator, X, y, cv=5, method='predict_proba')[:, 1], index=X.index)

    if not (X.index == y.index).all():
        warn('Indices are not aligned, maybe sort before?')

    y_pred = y_proba >= 0.50
    scores = score_clf_preds(y, y_pred)

    if report_coefs:
        cv_results = cross_validate(estimator, X, y, cv=5, return_estimator=True)
        fold_coefs = [e.coef_ for e in cv_results['estimator']]
        avg_coefs = np.mean(fold_coefs, axis=0)
        print(f'Selects {sum(avg_coefs != 0)} out of {X.shape[1]} features.')
        print(pd.Series(avg_coefs, X.columns).rename('coef'))

    return scores, y_proba


def score_clf_preds(y_true: pd.Series, y_pred: pd.Series, show_plot: bool = True) -> dict:
    """ Return commoon scoring functions calculated on response variable. """

    scoring_funcs = {
        'Acc': accuracy_score,
        'F1': f1_score
    }

    scores = {name: func(y_true, y_pred) for name, func in scoring_funcs.items()}

    if show_plot:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cfdf = pd.DataFrame({'pred_no': [tn, fn], 'pred_yes': [fp, tp]}, index=['actual_no', 'actual_yes'])
        display(cfdf.transform(lambda x: x / cfdf.sum().sum()))

        for name, score in scores.items():
            print(f'{name}: \t {score:.4}')

    return scores


def plot_prediction_bias_curve(y_true: pd.Series, y_pred_proba: pd.Series, num_bins: int = 100, ax=None) -> None:
    """

    Based on: https://developers.google.com/machine-learning/crash-course/classification/prediction-bias
    """

    if not ax:
        _, ax = plt.subplots()

    # Generate a sequence of N percentiles in the range [0,1]
    percentiles = np.linspace(1 / num_bins, 1, num_bins)

    # Find the respective value for each percentile to get equal sized bins
    # Take unique to avoid — ValueError: bins must be monotonically increasing or decreasing
    bin_edges = np.unique(np.quantile(y_pred_proba, percentiles))

    # Calculate the index of the bin to which each prediction belongs
    bin_memberships = np.digitize(y_pred_proba, bin_edges)

    # Calculate averages for predictions and actuals, then plot
    bins_mean_pred = [y_pred_proba[bin_memberships == i].mean() for i in range(len(bin_edges))]
    bins_mean_true = [y_true[bin_memberships == i].mean() for i in range(len(bin_edges))]
    ax.scatter(bins_mean_pred, bins_mean_true)

    # Print a diagnonal line for reference
    diag_min = min([min(bins_mean_pred), min(bins_mean_true)])
    diag_max = max([max(bins_mean_pred), max(bins_mean_true)])
    diag = [diag_min, diag_max]
    ax.plot(diag, diag, ls='--', c='.5')

    # Clean up the chart

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Label')
    ax.set_title('Prediction bias curve')


def plot_roc_curve(y_test: pd.Series, y_pred_prob: pd.Series, ax=None) -> None:

    if not ax:
        _, ax = plt.subplots()

    # Calculate and plot AUC cuve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    area = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = {:.3f})'.format(area))

    # Plot benchmarks
    ax.plot([0, 1], [0, 1], ls='--', c='.5', label='Random chance')
    ax.plot([0], [1], marker='o', markersize=6, color='green', label='Best possible model')
    ax.annotate('Optimal', xy=(0, 1), xytext=(0.01, 0.96), color='green')

    # Chart formatting
    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve')


def plot_precision_recall_curve(y_test: pd.Series, y_pred_prob: pd.Series, ax=None) -> None:

    if not ax:
        _, ax = plt.subplots()

    # Plot precision–recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    ax.step(recall, precision, color='darkblue', where='post', label='PR curve')

    # Plot benchmarks
    avg = y_test.mean()
    ax.plot([0, 1], [avg, avg], ls='--', c='.5', label='Random chance')
    ax.plot([1], [1], marker='o', markersize=6, color='green', label='Best possible model')
    ax.annotate('Optimal', xy=(1, 1), xytext=(0.88, 0.95), color='green')

    # Chart formatting
    ax.legend(loc='lower right')
    ax.set_xlabel('Recall (TPR)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_ylim([0, 1.05])
    ax.set_title('Precision vs. Recall curve')


def plot_clf_diagnostics(y_true: pd.Series, y_pred: pd.Series) -> None:
    """ Generate a side-by-side plot of prediction–bias curve, ROC curve, and PR curve """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    ax1, ax2, ax3 = axes

    plot_prediction_bias_curve(y_true, y_pred, ax=ax1)
    plot_roc_curve(y_true, y_pred, ax=ax2)
    plot_precision_recall_curve(y_true, y_pred, ax=ax3)

    plt.subplots_adjust(wspace=0.3)


if __name__ == '__main__':
    pass
