import os
import numpy as np
import numpy.random as rnd
import pandas as pd
#import shap
from joblib import Parallel, delayed
from doubly_robust import doubly_robust
from sklearn.decomposition import PCA
from simulate_behavioural_scores import simulate_behavioural_scores_single
from simulate_behavioural_scores import simulate_behavioural_scores_AND
from simulate_behavioural_scores import simulate_behavioural_scores_OR
from simulate_behavioural_scores import simulate_behavioural_scores_SUM
from simulate_behavioural_scores import simulate_behavioural_scores_XOR
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import desparsified_lasso as dsl
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from doubly_robust import doubly_robust
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# TODO : Docstring functions
# TODO : depreciate cname argument except in fit_one_region


def fit_one_region(X, y, cname, model='BART', lesion_threshold=0.6):

    '''Fits a causal model for a single brain region.

    Inputs
    ------
    X : 2D array of shape (n_subjects, n_regions)
        Contains the lesion status of the different
        brain regions from the lesion map
    y : list or 1D array of shape (n_subjects)
        Vector of behavioural scores
    cname : int
        Index of the region to use as a treatment variable
        Note : The index should be relative to the array, and
        not to the name of the region in the atlas
    model : str
        Causal model to use to perform inference.
        Valid values are "BART" and "DR"
    lesion_threshold : float
        Proportion of lesioned voxels above which a region
        is considered lesioned.
        Used to binarize the treatment variable.
        Does nothing if X is output of cull_dataset function.

    Outputs
    -------
    tau : array of shape (2,) if model == BART, float if model == DR
        The estimated average treatment effect of the region
        denoted by cname.

    Notes
    -----
    This needs to be its own function because we need to redo the imports
    at the start of the function for each parallel thread, 
    in case the model is BART'''

    if model == 'BART':
        # rpy2 imports
        import rpy2
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()  # This needs to be called
        brc = importr('bartCause')

    # Create treatment assignment vector
    W_temp = X[:, cname]
    W_temp[np.where(W_temp >= lesion_threshold)] = 1
    W_temp[np.where(W_temp < lesion_threshold)] = 0
    W_temp = W_temp.astype(int)

    # Checking if there are both treated and untreated in sample
    # Does nothing if X is output of cull_dataset function.
    X_temp = np.delete(X, cname, axis=1)
    if model == 'BART':
        # PCA X
        reductor = PCA(n_components=20)
        X_temp = reductor.fit_transform(X_temp)
        bart_model = brc.bartc(y, W_temp, X_temp, method_rsp='p.weight',
                               estimand='ate')

        tau = brc.summary_bartcFit(bart_model).rx2('estimates')
        return tau[0]

    elif model == 'DR':
        tau = doubly_robust(X_temp, y, W_temp, n_jobs=1, RF=True, cv=None)
        return tau

    else:
        return NotImplementedError('model must be either "BART" or "DR"')


def get_coefs(X, y, model, cnames, lesion_threshold=0.6, n_jobs=30):
    ''' Gets coefs for each model.

    Inputs
    ------
    X : Pandas DataFrame of shape (n_subjects, n_regions)
        Contains the lesion status of the different
        brain regions from the lesion map
    y : list or 1D array of shape (n_subjects)
        Vector of behavioural scores
    model : str
        Model to use to perform inference.
        Valid values are "BART", "DR", "SVR", "DLASSO",
        "RF", "RF+SHAP"
    lesion_threshold : float
        Proportion of lesioned voxels above which a region
        is considered lesioned.
        Used to binarize the treatment variable in the case of causal models.
        Does nothing if X is output of cull_dataset function.
    n_jobs
        Number of threads to use in parallel computing.
        Passed to joblib.Parallel.

    Outputs
    -------
    coefs : 1D array of shape (n_regions,)
        Contains the model weights, feature importance or
        average treatment effects for each region
    sigmas : 1D array of shape (n_regions,)
        Contains the estimated standard deviations on model weights/
        average treatment effects for each region if model == "BART"
        or model == "DLASSO"'''

    if model == 'BART':
        coefs = []
        sigmas = []
        BART_estimates = Parallel(n_jobs=n_jobs, verbose=10)(delayed(fit_one_region)
                                        (X, y, cname, 'BART', lesion_threshold)
                                        for cname in range(X.shape[1]))
        for tau in BART_estimates:
            coefs.append(tau[0])
            sigmas.append(tau[1])

        coefs, sigmas = np.array(coefs), np.array(sigmas)

        return coefs, sigmas

    elif model == 'DR':
        coefs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(fit_one_region)
                                            (X, y, cname, 'DR', lesion_threshold)
                                            for cname in range(X.shape[1]))
        coefs = np.array(coefs)
        return coefs

    elif model == 'DLASSO':
        coefs, sigmas = dsl.desparsified_lasso_std_err(X, y,
                                                       max_iter=10000, c=0.1)

        return coefs, sigmas

    elif model == 'SVR':
        estimator = LinearSVR()
        param_grid = np.array([10 ** k for k in np.arange(-2, 3, dtype=float)])
        param_grid = {'epsilon': param_grid}
        grid_search = GridSearchCV(estimator, param_grid)
        grid_search.fit(X, y)
        best_SVR = grid_search.best_estimator_
        best_SVR.fit(X, y)
        coefs = best_SVR.coef_

        return coefs

    elif model == "RF":
        estimator = RandomForestRegressor()
        param_grid = np.array([10 ** k for k in np.arange(2, 4)])
        param_grid = {'n_estimators': param_grid}
        grid_search = GridSearchCV(estimator, param_grid)
        grid_search.fit(X, y)
        best_RF = grid_search.best_estimator_
        best_RF.fit(X, y)
        feature_importances = permutation_importance(best_RF, X, y)
        coefs = feature_importances['importances_mean']

        return coefs

    elif model == 'RF+SHAP':
        estimator = RandomForestRegressor()
        param_grid = np.array([10 ** k for k in np.arange(2, 4)])
        param_grid = {'n_estimators': param_grid}
        grid_search = GridSearchCV(estimator, param_grid)
        grid_search.fit(X, y)
        best_RF = grid_search.best_estimator_
        best_RF.fit(X, y)

        explainer = shap.TreeExplainer(best_RF)
        shap_values = explainer.shap_values(X)
        coefs = np.mean(shap_values, axis=0)

        return coefs
    else:
        return NotImplementedError('Model must be either "BART", "DR",'
                                   ' "DLASSO", "SVR", "RF", "RF+SHAP" ')


def get_zscores(X, coefs, sigmas=None, fpath=None, fname=None,
                save_results=False):
    '''Compute zscores from model coefficients

    Inputs
    ------
    X : Pandas DataFrame of shape (n_subjects, n_regions)
        Contains the lesion status of the different
        brain regions from the lesion map
    coefs : list or 1D array of shape (n_regions)
        Model coefficients for each brain region
    sigmas : list or 1D array of shape (n_regions), optional
       Standard deviations on model coefficients.
       Only provided by BART and DLASSO models.
       Leave the default value (None) if unsure.
    fpath : str
        Path to the directory in which to save zscores
    fname : str
        Name of the file in which to save zscores
    save_results : bool
        Set to true to save results to disk

    Outputs
    -------
    zscores : 1D array of shape (n_regions,)
        Contains the zscores for each brain region'''

    zscores = np.ones(X.shape[1])

    # If the model doesn't provide standard deviation estimates
    # robustly fit a gaussian density and compute
    # pseudo-zscores
    if sigmas is None:
        mu = np.median(coefs)
        s = np.mean(np.abs(coefs - mu))
        if s == 0:
            zscores = np.zeros(X.shape[1])
        else:
            for i, x in enumerate(coefs):
                z_score = (x - mu) / s
                zscores[i] = z_score

    else:
        for i, x in enumerate(coefs):
            z_score = x / sigmas[i]
            zscores[i] = z_score
    zscores = pd.DataFrame(zscores)
    zscores.set_index(X.columns, inplace=True)

    if save_results:
        zscores.to_pickle(os.path.join(fpath, fname))

    return zscores


def make_AUCs(X, zscores, scenario='single', rois=[100, 101]):
    '''Computes AUCs from zscores

    Inputs
    ------
    X : Pandas DataFrame of shape (n_subjects, n_regions)
        Contains the lesion status of the different
        brain regions from the lesion map
    zscores : list or 1D array of shape (n_regions)
        Zscores for each brain region
    scenario : str
       Lesion-behaviour interaction simulation scenario
       under which the models were fitted.
       Valid values are "single", "OR", "AND", "SUM", "XOR"
    rois : list of length 2
        ROI pair which were used to simulate behavioural scores.

    Outputs
    -------
    AUC : float
        AUC of the precision-recall curve obtained
        from the Zscores.'''

    # Get the ROIs to detect according to the scenario
    switcher_rois = {'single': [rois[1]],
                     'OR': rois,
                     'AND': rois,
                     'SUM': rois,
                     'XOR': rois}
    rois_to_detect = switcher_rois.get(scenario)

    # Construct target vector
    # And try to avoid indexing problems as well
    y_true = np.zeros(X.shape[1])
    y_true = pd.DataFrame(y_true)
    y_true.set_index(X.columns, inplace=True)
    for roi in rois_to_detect:
        y_true.loc[roi] = 1
    y_true = y_true.to_numpy()

    # Convert zscores back to numpy array
    zscores = zscores.to_numpy()

    precision, recall, _ = precision_recall_curve(y_true=y_true,
                                                  probas_pred=zscores)
    AUC = auc(recall, precision)

    return AUC


def check_bootstrap_sample_is_valid(X_bs, rois, min_cond_size=4):
    ''' Checks if a bootstrap sample is valid for inference, by checking
        if there are enough subjects
        with a lesion in one ROI, both ROIs,
        and both versions of the OR case.

    Inputs
    ------
    X_bs : Pandas DataFrame of shape (bs_size, n_regions)
        Contains the lesion status of the different
        brain regions from the lesion map in the bootstrap
        sample to check.
    rois : list of length 2
        ROI pair which were used to simulate behavioural scores.
    min_cond_size : int
        The minimum number of samples for each case among :
        lesions in neither ROIs, lesion in ROI 1 but not ROI 2,
        lesion in ROI 2 but not ROI 1 and lesion in both ROIs.
        Must be at least 1. Too large values may cause nearly all
        bootstrap samples to be rejected, increasing computation time.

    Outputs
    -------
    bs_sample_is_valid : bool
        Whether the bootstrap sample passed as input is valid
        for inference.'''
    
    lesioned_0 = X_bs[rois[0]] == 1
    lesioned_1 = X_bs[rois[1]] == 1
    not_lesioned_0 = X_bs[rois[0]] == 0
    not_lesioned_1 = X_bs[rois[1]] == 0

    num_both_rois = len(X_bs[lesioned_0 & lesioned_1])
    num_one_roi_0 = len(X_bs[lesioned_0 & not_lesioned_1])
    num_one_roi_1 = len(X_bs[not_lesioned_0 & lesioned_1])
    num_no_roi = len(X_bs[not_lesioned_0 & not_lesioned_1])

    useless_regions = (X_bs == 0).all()

    if (num_both_rois >= min_cond_size and num_one_roi_0 >= min_cond_size
        and num_one_roi_1 >= min_cond_size and num_no_roi >= min_cond_size
            and len(X_bs.columns[useless_regions]) == 0):

        return True
    else:
        return False


def bootstrap_AUCs(X, model, SNR=1, n_bs=50, bs_size=150, rois=[100, 101],
                   min_cond_size=4, scenario='single', lesion_threshold=0.6,
                   n_jobs=30):
    ''' Computes AUCs through n_bs bootstrap runs. This is the function you want to
        use to replicate my experiments.

    Inputs
    ------
    X : Pandas DataFrame of shape (n_subjects, n_regions)
        Contains the lesion status of the different
        brain regions from the lesion map
    model : str
        Model to use to perform inference.
        Valid values are "BART", "DR", "SVR", "DLASSO",
        "RF", "RF+SHAP"
    n_bs : int
        Number of bootstrap runs to perform
    bs_size : int
        Size of each bootstrap sample.
    rois : list of length 2
        ROI pair which to use when simulating behavioural scores.
    min_cond_size : int
        Used to check for bootstrap sample validity.
        The minimum number of samples for each case among :
        lesions in neither ROIs, lesion in ROI 1 but not ROI 2,
        lesion in ROI 2 but not ROI 1 and lesion in both ROIs.
        Must be at least 1. Too large values may cause nearly all
        bootstrap samples to be rejected, increasing computation time.
    scenario : str
        Scenario under which to simulate behavioural scores.
        Valid values are "single", "OR", "AND", "SUM"
    lesion_threshold : float
        Proportion of lesioned voxels above which a region
        is considered lesioned.
        Used to binarize the treatment variable in the case of causal models.
        Does nothing if X is output of cull_dataset function.
    n_jobs
        Number of threads to use in parallel computing.
        Passed to joblib.Parallel.

    Outputs
    -------
    AUC_array : 1D array of shape (n_bs,)
        AUC of the precision-recall curve
        obtained for each bootstrap run'''

    AUC_array = np.zeros(n_bs)
    cnames = np.array(list(X.iteritems()))[:, 0]
    # Make switcher dict of cases
    switcher = {'single': simulate_behavioural_scores_single,
                'OR': simulate_behavioural_scores_OR,
                'AND': simulate_behavioural_scores_AND,
                'SUM': simulate_behavioural_scores_SUM,
                'XOR': simulate_behavioural_scores_XOR}
    # Grab the right simulation function
    simulation_function = switcher.get(scenario)
    noise_level = 1 / SNR

    # Construct the bootstrap samples first
    X_bootstrapped_array = np.empty(shape=(n_bs, bs_size, X.shape[1]))
    y_bootstrapped_array = np.empty(shape=(n_bs, bs_size))

    for j in range(n_bs):
        bs_sample_is_valid = False
        # Boostrap sample from X and y
        while not bs_sample_is_valid:
            bs_indices = rnd.choice(np.arange(131), size=bs_size)
            X_bs = X.iloc[bs_indices]
            # Check if boostrap sample is valid
            bs_sample_is_valid = check_bootstrap_sample_is_valid(X_bs,
                                                                 rois,
                                                                 min_cond_size)
            if bs_sample_is_valid:
                # Proceed with simulating bootstrap outcomes
                if scenario == 'single':
                    y_bs = simulation_function(X_bs, roi=rois[1], amplitude=1,
                                               noise_level=noise_level)
                elif scenario == 'OR' or scenario == 'AND' or scenario =='XOR':
                    y_bs = simulation_function(X_bs, rois=rois, amplitude=1,
                                               noise_level=noise_level)
                elif scenario == 'SUM':
                    y_bs = simulation_function(X_bs, rois=rois, amplitudes=[1 for i in rois],
                                               noise_level=noise_level)
                else:
                    raise ValueError('Invalid scenario choice')

                X_bootstrapped_array[j] = X_bs
                y_bootstrapped_array[j] = y_bs
                print('%s th bootstrap sample selected' % j)

    # Now that we have a valid bootstrap sample, calculate AUCs
    # And use joblib to go parallelize
    # Make a separate case for BART and DR to avoid worker oversubscription
    if model == 'BART':
        results_array = []
        # Use a for loop instead of parallelization
        for a, b in zip(X_bootstrapped_array, y_bootstrapped_array):
            results_array.append(get_coefs(X=a, y=b, model=model,
                                           cnames=cnames,
                                           lesion_threshold=lesion_threshold,
                                           n_jobs=n_jobs))
        results_array = np.array(results_array)
        coefs_array = results_array[:, 0]
        sigmas_array = results_array[:, 1]

    elif model == 'DR':
        coefs_array = []
        for a, b in zip(X_bootstrapped_array, y_bootstrapped_array):
            coefs_array.append(get_coefs(X=a, y=b, model=model,
                                         cnames=cnames,
                                         lesion_threshold=lesion_threshold,
                                         n_jobs=n_jobs))
        coefs_array = np.array(coefs_array)

    elif model == 'DLASSO':
        results_array = Parallel(n_jobs=n_jobs, verbose=10)(delayed(get_coefs)(
            X=a,
            y=b,
            model=model,
            cnames=cnames,
            lesion_threshold=lesion_threshold,
            n_jobs=n_jobs)
            for a, b in zip(X_bootstrapped_array, y_bootstrapped_array)
            )
        results_array = np.array(results_array)

        coefs_array = results_array[:, 0]
        sigmas_array = results_array[:, 1]

    else:
        coefs_array = Parallel(n_jobs=n_jobs, verbose=10)(delayed(get_coefs)(
            X=a,
            y=b,
            model=model,
            cnames=cnames,
            lesion_threshold=lesion_threshold,
            n_jobs=n_jobs)
            for a, b in zip(X_bootstrapped_array, y_bootstrapped_array)
            )
        coefs_array = np.array(coefs_array)

    # Compute zscores and AUCs
    if model == 'BART' or model == 'DLASSO':
        for i in range(n_bs):
            coefs = coefs_array[i]
            sigmas = sigmas_array[i]
            zscores = get_zscores(X, coefs, sigmas)
            AUC = make_AUCs(X, zscores, scenario, rois)
            AUC_array[i] = AUC
    else:
        for i in range(n_bs):
            coefs = coefs_array[i]
            zscores = get_zscores(X, coefs)
            AUC = make_AUCs(X, zscores, scenario, rois)
            AUC_array[i] = AUC

    return AUC_array
