from typing import Tuple
from warnings import warn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline


mpl.style.use('default')
sns.set_style('white')


def plot_corr_triangle(df: pd.DataFrame):
    """ Plot a heatmap to visualize the correlation between variables.
        Remove the upper triangle for easier interpretability, because it is duplicate info. """
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 6))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    plt.title('Correlation of predictor variables')


def check_assumptions_for_linear_model(X: pd.DataFrame) -> pd.DataFrame:
    """ Asserts that a DataFrame has no null values or infinite values. """

    disallowed_dtypes = ['datetime', 'object']
    for dtype in disallowed_dtypes:
        disallowed_columns = X.select_dtypes(dtype).columns
        assert len(disallowed_columns) == 0, f'Columns of dtype {dtype}: {", ".join(disallowed_columns)}'
        
    cols_with_inf_values = X.apply(np.isinf).any().loc[lambda x: x == True].index
    assert len(cols_with_inf_values) == 0, 'Columns with infinite values:' + '\n\t' + '\n\t'.join(cols_with_inf_values)
    print('[✔] No columns with infinite values')

    cols_with_null_values = X.isnull().any().loc[lambda x: x == True].index
    assert len(cols_with_null_values) == 0, 'Columns with null values:' + '\n\t' + '\n\t'.join(cols_with_null_values)
    print('[✔] No columns with null values')

    return X


def eval_baseline(X, y, plot=False, verbose=False):
    """ Fit a linear regression with only the intercept to calculate baseline metrics. """
    intercept = pd.DataFrame(np.ones_like(y), index=X.index)
    scores, preds = fit_eval_reg_with_CV(LinearRegression(), intercept, y)

    if verbose:
        avg_incr_gbv = y.mean()
        print(f'\nAverage incremental GBV:\t€ {avg_incr_gbv:.4}')
        print(f'Average mean from CV folds:\t€ {preds.mean():.4}')

    if plot:
        plot_diagnostics(y, preds, bin_width=20)

    return scores, preds


def fit_eval_reg_with_CV(estimator, X: pd.DataFrame, y: pd.Series, cv=5, report_coefs=False) -> Tuple[dict, pd.Series]:
    """ Fit and evaluate a regression model using CV to obtain realistic estimates of performance.

    Returns:
        scores (dict): Dict containing common evaluation metrics computed on predictions
        preds (pd.Series): Raw predictions for each observation

    """

    if not (X.index == y.index).all():
        warn('Indices are not aligned, maybe sort before?')

    preds = pd.Series(cross_val_predict(estimator, X, y, cv=cv), index=X.index)
    scores = score_reg_preds(y, preds)

    if report_coefs:
        print('')
        cv_results = cross_validate(estimator, X, y, cv=5, return_estimator=True)
        if isinstance(cv_results['estimator'][0], Pipeline):
            avg_coefs = np.mean([pipe.steps[-1][1].coef_ for pipe in cv_results['estimator']], axis=0)
        else:
            avg_coefs = np.mean([e.coef_ for e in cv_results['estimator']], axis=0)

        print(f'Selects {sum(avg_coefs != 0)} out of {X.shape[1]} features.')
        print(pd.Series(avg_coefs, X.columns).rename('coef').pipe(n_largest_coefs))

    return scores, preds


def score_reg_preds(y_true, y_pred, print_output=True) -> dict:
    """ Calculate multiple scoring functions for regression predictions. """
    scoring_funcs = {
        'R^2': r2_score,
        'RMSE': lambda X, y: mean_squared_error(X, y) ** 0.5,
        'MAE': mean_absolute_error
    }

    scores = {name: func(y_true, y_pred) for name, func in scoring_funcs.items()}

    if print_output:
        for name, score in scores.items():
            print(f'{name}: \t {score:.4}')

    return scores


def plot_diagnostics(y_true: pd.Series, y_pred: pd.Series, obs_max_quantile=0.99, bin_width=None) -> None:
    """ Visualize predictions vs. actuals and residuals plot side-by-side """
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    assert (y_pred.index == y_true.index).all(), 'Indices are not aligned.'

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
    ax1, ax2 = axes

    ax1.set_title('Predictions vs. Actuals')
    y_true_clipped = y_true.clip(upper=y_true.quantile(obs_max_quantile))
    y_pred_clipped = y_pred.clip(upper=y_true.quantile(obs_max_quantile))

    if bin_width:
        bins = np.arange(min(y_true_clipped.min(), y_pred_clipped.min()), max(y_true_clipped.max(), y_pred_clipped.max()), bin_width)
    else:
        bins = None

    sns.distplot(y_true_clipped, kde=False, bins=bins, ax=ax1, label='Actuals')
    sns.distplot(y_pred_clipped, kde=False, bins=bins, ax=ax1, label='Predictions')

    ax1.legend()
    ax1.set_xlabel('')
    ax1.get_yaxis().set_visible(False)

    ax2.set_title('Residuals (error)')
    resids = y_pred - y_true
    sns.distplot(resids.clip(lower=resids.quantile(0.005), upper=resids.quantile(0.995)), ax=ax2, kde=False, color='pink', label='Residuals')
    ax2.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


def find_best_alpha(X, y):
    """ Show cross-validated error across various possible values of alpha

    Adapted from: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html """

    import time

    # This is to avoid division by zero while doing np.log10
    EPSILON = 1e-10

    # Compute paths
    print("Computing regularization path using the coordinate descent lasso...")
    t1 = time.time()
    model = LassoCV(cv=20, alphas=np.geomspace(4, 0.0000001, 10)).fit(X, y)
    t_lasso_cv = time.time() - t1

    # Display results
    m_log_alphas = -np.log10(model.alphas_ + EPSILON)

    plt.figure()
    # ymin, ymax = 2300, 3800
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_ + EPSILON), linestyle='--', color='k',
                label='alpha: CV estimate')

    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('MSE for each fold at various alphas '
              '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')
    print(f'Best alpha: {model.alpha_:.4}')
    return model


def sklearn_lasso_coefs_plot(X, y):
    """ Show the path taken by coefficients as Lasso shrinks them towards zero.

    Adapted from: https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html """

    from itertools import cycle
    from sklearn.linear_model import lasso_path

    eps = 5e-3  # the smaller it is the longer is the path

    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=True)

    # Display results

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef, label, c in zip(coefs_lasso, X.columns, colors):
        _ = plt.plot(neg_log_alphas_lasso, coef, label=label, c=c)
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso path')
    plt.legend(loc='upper left')
    plt.axis('tight')

    plt.show()


def n_largest_coefs(coefs: pd.Series, n=10) -> pd.Series:
    """ Return the n largest absolute values from a pandas Series """
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(n)


if __name__ == '__main__':
    pass
