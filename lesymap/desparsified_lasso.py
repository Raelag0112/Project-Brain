import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model.coordinate_descent import _alpha_grid
import scipy.stats as st


# %cd python/code/hidimstat/hidimstat


import numpy as np
from numpy.linalg import inv, norm
from sklearn.utils import resample
from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsCV
from sklearn.model_selection import LeaveOneGroupOut


def reid(X, y, method="lars", tol=1e-6, max_iter=1e+3):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        method : string, optional
            The method for the CV-lasso: "lars" or "lasso"
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if int(max_iter / 5) <= n_features:
        max_iter = n_features * 5

    if method == "lars":
        clf_lars_cv = LassoLarsCV(max_iter=max_iter, normalize=False, cv=3)
        clf_lars_cv.fit(X, y)
        error = clf_lars_cv.predict(X) - y
        support = sum(clf_lars_cv.coef_ != 0)

    elif method == "lasso":
        clf_lasso_cv = LassoCV(tol=tol, max_iter=max_iter, cv=3)
        clf_lasso_cv.fit(X, y)
        error = clf_lasso_cv.predict(X) - y
        support = sum(clf_lasso_cv.coef_ != 0)

    sigma_hat = np.sqrt((1. / (n_samples - support)) * norm(error) ** 2)

    return sigma_hat


def desparsified_lasso(X, y, max_iter=5000, tol=1e-3, method="lasso", c=0.01):
    """Desparsified Lasso

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        method : string, optional
            The method for the nodewise lasso: "lasso", "lasso_cv" or
            "zhang_zhang"
        c : float, optional
            Only used if method="lasso". Then alpha = c * alpha_max.
        """

    X = np.asarray(X)

    n_samples, n_features = X.shape

    Z = np.zeros((n_samples, n_features))

    if method == "lasso":

        Gram = np.dot(X.T, X)

        k = c * (1. / n_samples)
        alpha = k * np.max(np.abs(Gram - np.diag(np.diag(Gram))), axis=0)

    elif method == "lasso_cv":

        clf_lasso_loc = LassoCV(max_iter=max_iter, tol=tol, cv=3)

    # Calculating Omega Matrix i = 0
    for i in range(n_features):

        if method == "lasso":

            # Gram_loc = np.delete(np.delete(Gram, obj=i, axis=0),
            #                      obj=i, axis=1)
            # clf_lasso_loc = Lasso(alpha=alpha[i], precompute=Gram_loc,
            #                       tol=tol)
            clf_lasso_loc = Lasso(alpha=alpha[i], max_iter=max_iter, tol=tol)

        if method == "lasso" or method == "lasso_cv":

            X_new = np.delete(X, i, axis=1)
            clf_lasso_loc.fit(X_new, X[:, i])

            Z[:, i] = X[:, i] - clf_lasso_loc.predict(X_new)

        elif method == "zhang_zhang":

            print("i = ", i)
            X_new = np.delete(X, i, axis=1)
            alpha, z, eta, tau = lpde_regularizer(X_new, X[:, i])

            Z[:, i] = z

    # Lasso regression
    clf_lasso_cv = LassoCV(cv=3)
    clf_lasso_cv.fit(X, y)
    beta_lasso = clf_lasso_cv.coef_

    # Estimating the coefficient vector
    beta_bias = y.T.dot(Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    return beta_hat


def desparsified_lasso_confint(X, y, confidence=0.95, max_iter=5000,
                               tol=1e-3, method="lasso", c=0.01):
    """Desparsified Lasso with confidence intervals

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        confidence : float, optional
            Confidence level used to compute the confidence intervals.
            Each value should be in the range [0, 1].
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        method : string, optional
            The method for the nodewise lasso: "lasso", "lasso_cv" or
            "zhang_zhang"
        c : float, optional
            Only used if method="lasso". Then alpha = c * alpha_max.
        """

    X = np.asarray(X)

    n_samples, n_features = X.shape

    Z = np.zeros((n_samples, n_features))
    omega_diag = np.zeros(n_features)
    omega_invsqrt_diag = np.zeros(n_features)

    quantile = st.norm.ppf(1 - (1 - confidence) / 2)

    if method == "lasso":

        Gram = np.dot(X.T, X)

        k = c * (1. / n_samples)
        alpha = k * np.max(np.abs(Gram - np.diag(np.diag(Gram))), axis=0)

    elif method == "lasso_cv":

        clf_lasso_loc = LassoCV(max_iter=max_iter, tol=tol, cv=3)

    # Calculating Omega Matrix
    for i in range(n_features):

        if method == "lasso":

            Gram_loc = np.delete(np.delete(Gram, obj=i, axis=0), obj=i, axis=1)
            clf_lasso_loc = Lasso(alpha=alpha[i], precompute=Gram_loc,
                                  max_iter=max_iter, tol=tol)

        if method == "lasso" or method == "lasso_cv":

            X_new = np.delete(X, i, axis=1)
            clf_lasso_loc.fit(X_new, X[:, i])

            Z[:, i] = X[:, i] - clf_lasso_loc.predict(X_new)

        elif method == "zhang_zhang":

            print("i = ", i)
            X_new = np.delete(X, i, axis=1)
            alpha, z, eta, tau = lpde_regularizer(X_new, X[:, i])

            Z[:, i] = z

        omega_diag[i] = (n_samples * np.sum(Z[:, i] ** 2) /
                         np.sum(Z[:, i] * X[:, i]) ** 2)

    omega_invsqrt_diag = omega_diag ** (-0.5)

    # Lasso regression
    clf_lasso_cv = LassoCV(cv=3)
    clf_lasso_cv.fit(X, y)
    beta_lasso = clf_lasso_cv.coef_

    # Estimating the coefficient vector
    beta_bias = y.T.dot(Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    sigma_hat = reid(X, y)

    confint_radius = np.abs(quantile * sigma_hat /
                            (np.sqrt(n_samples) * omega_invsqrt_diag))
    cb_max = beta_hat + confint_radius
    cb_min = beta_hat - confint_radius

    return beta_hat, cb_min, cb_max


def desparsified_lasso_std_err(X, y, Z=None, sigma=None, max_iter=5000,
                               tol=1e-3, method="lasso", c=0.01, groups=None,
                               return_Z=False):
    """Desparsified Lasso with confidence intervals

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        method : string, optional
            The method for the nodewise lasso: "lasso", "lasso_cv" or
            "zhang_zhang"
        c : float, optional
            Only used if method="lasso". Then alpha = c * alpha_max.
        """

    X = np.asarray(X)

    n_samples, n_features = X.shape

    se_hat = np.zeros(n_features)

    if sigma is None:
        sigma_hat = reid(X, y)
    else:
        sigma_hat = sigma

    if Z is None:

        Z = np.zeros((n_samples, n_features))

        if method == "lasso":

            Gram = np.dot(X.T, X)

            k = c * (1. / n_samples)
            alpha = k * np.max(np.abs(Gram - np.diag(np.diag(Gram))), axis=0)

        elif method == "lasso_cv":

            clf_lasso_loc = LassoCV(max_iter=max_iter, tol=tol, cv=3)

        # Calculating Omega Matrix
        for i in range(n_features):

            if method == "lasso":

                Gram_loc = \
                    np.delete(np.delete(Gram, obj=i, axis=0), obj=i, axis=1)
                clf_lasso_loc = \
                    Lasso(alpha=alpha[i], precompute=Gram_loc,
                          max_iter=max_iter, tol=tol)

            if method == "lasso" or method == "lasso_cv":

                X_new = np.delete(X, i, axis=1)
                clf_lasso_loc.fit(X_new, X[:, i])

                Z[:, i] = X[:, i] - clf_lasso_loc.predict(X_new)

            elif method == "zhang_zhang":

                print("i = ", i)
                X_new = np.delete(X, i, axis=1)
                alpha, z, eta, tau = lpde_regularizer(X_new, X[:, i])

                Z[:, i] = z

    for i in range(n_features):
        se_hat[i] = (sigma_hat * np.linalg.norm(Z[:, i]) /
                     np.abs(np.sum(Z[:, i] * X[:, i])))

    # Lasso regression
    clf_lasso_cv = LassoCV(cv=3)
    clf_lasso_cv.fit(X, y)
    beta_lasso = clf_lasso_cv.coef_

    # Estimating the coefficient vector
    beta_bias = y.T.dot(Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    if return_Z:
        return beta_hat, se_hat, Z

    return beta_hat, se_hat


def lpde_regularizer(X, y, grid=100, alpha_max=None, kappa_0=0.25,
                     kappa_1=0.5, c_max=0.99, eps=1e-3):

    X = np.asarray(X)
    n_samples, n_features = X.shape

    eta_star = np.sqrt(2 * np.log(n_features))

    z_grid = np.zeros(grid * n_samples).reshape(grid, n_samples)
    eta_grid = np.zeros(grid)
    tau_grid = np.zeros(grid)

    if alpha_max is None:
        alpha_max = np.max(np.dot(X.T, y)) / n_samples

    alpha_0 = eps * c_max * alpha_max
    z_grid[0, :], eta_grid[0], tau_grid[0] = lpde_regularizer_substep(X, y,
                                                                      alpha_0)

    if eta_grid[0] > eta_star:
        eta_star = (1 + kappa_1) * eta_grid[0]

    alpha_1 = c_max * alpha_max
    z_grid[-1, :], eta_grid[-1], tau_grid[-1] = (lpde_regularizer_substep(X, y,
                                                 alpha_1))

    alpha_grid = _alpha_grid(X, y, eps=eps, n_alphas=grid)[::-1]
    alpha_grid[0] = alpha_0
    alpha_grid[-1] = alpha_1
    # alpha_grid = np.logspace(np.log10(alpha_0), np.log10(alpha_1), num=grid)
    # alpha_grid = np.linspace(alpha_0, alpha_1, num=grid)
    for i, alpha in enumerate(alpha_grid[1:-1], 1):
        z_grid[:, i], eta_grid[i], tau_grid[i] = (lpde_regularizer_substep(X,
                                                  y, alpha))

    # tol_factor must be inferior to (1 - 1 / (1 + kappa_1)) = 1 / 3 (default)
    index_1 = (grid - 1) - (eta_grid <= eta_star)[-1].argmax()

    tau_star = (1 + kappa_0) * tau_grid[index_1]

    index_2 = (tau_grid <= tau_star).argmax()

    return (alpha_grid[index_2], z_grid[:, index_2], eta_grid[index_2],
            tau_grid[index_2])


def lpde_regularizer_substep(X, y, alpha):

    clf_lasso = Lasso(alpha=alpha)
    clf_lasso.fit(X, y)

    z = y - clf_lasso.predict(X)
    z_norm = np.linalg.norm(z)
    eta = np.max(np.dot(X.T, z)) / z_norm
    tau = z_norm / np.sum(y * z)

    return z, eta, tau
