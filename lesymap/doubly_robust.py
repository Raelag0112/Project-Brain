import numpy as np
import numpy.random as rnd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV


class DoublyRobustEstimator_ITE:

    def __init__(self, propensity_threshold=0.01,
                 n_trees=1000, max_depth=None, n_jobs=10):
        self.propensity_threshold = propensity_threshold
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        
    def fit(self, X, Y, W):
        # Create and fit propensity score estimator
        self.propensity_regressor = LogisticRegression(solver='lbfgs')
        self.calibrated_propensity_regressor = CalibratedClassifierCV(
                                    base_estimator=self.propensity_regressor,
                                    cv=5)
        self.calibrated_propensity_regressor.fit(X, W)

        # Create and fit conditional mean regressors
        # self.mu_0 = RandomForestRegressor(n_estimators=self.n_trees,
        #                                   max_depth=self.max_depth)
        # self.mu_1 = RandomForestRegressor(n_estimators=self.n_trees,
        #                                   max_depth=self.max_depth)
        self.mu = RandomForestRegressor(n_estimators=self.n_trees,
                                        max_depth=self.max_depth,
                                        n_jobs=self.n_jobs)
        self.mu_0 = RidgeCV()
        self.mu_1 = RidgeCV()
        
        # Find indexes of treated and contol subjects
        treated_index = np.where(W == 1)[0]
        control_index = np.where(W == 0)[0]

        self.mu_0.fit(X[control_index], Y[control_index])
        self.mu_1.fit(X[treated_index], Y[treated_index])


        aggregate = np.concatenate((X, W.reshape(len(W), 1)), axis=1)
        self.mu.fit(aggregate, Y)

    def predict(self, X, Y, W):
        # Find indexes of treated and contol subjects
        treated_index = np.where(W == 1)[0]
        control_index = np.where(W == 0)[0]
        
        estimated_propensity = self.calibrated_propensity_regressor.predict_proba(
                                                            X)[:, 1]
        
        # Threshold estimated propensities
        if self.propensity_threshold != 0:
            estimated_propensity[estimated_propensity >
                                 (1 - self.propensity_threshold)
                                 ] = (1 - self.propensity_threshold)

            estimated_propensity[estimated_propensity <
                                 self.propensity_threshold
                                 ] = self.propensity_threshold
        estimated_mu_0 = self.mu_0.predict(X)
        estimated_mu_1 = self.mu_1.predict(X)


        agg_1 = np.hstack((X, np.ones((len(X), 1))))
        bla_1 = self.mu.predict(agg_1)
        agg_0 = np.hstack((X, np.zeros((len(X), 1))))
        bla_0 = self.mu.predict(agg_0)





        inverse_propensity_treated = (W / estimated_propensity
                                      * (Y - bla_1))
        inverse_propensity_control = ((1 - W) / (1 - estimated_propensity)
                                      * (Y - bla_0))
        regression_term = bla_1 - bla_0

        result = (inverse_propensity_treated
                  - inverse_propensity_control
                  + regression_term)

        return result


def doubly_robust(X, Y, W, RF=True,
                  propensity_threshold=0.01,
                  n_trees=1000, cv=5, max_depth=None,
                  n_jobs=10):
    '''Doubly robust estimator for ATE'''

    # Fit propensity regressor
    propensity_regressor = LogisticRegression(solver='lbfgs')
    if cv is None:
        propensity_regressor.fit(X, W)
        estimated_propensity = propensity_regressor.predict_proba(
                                                            X)[:, 1]
    else:
        calibrated_propensity_regressor = CalibratedClassifierCV(
                                        base_estimator=propensity_regressor,
                                        cv=cv)
        calibrated_propensity_regressor.fit(X, W)

        estimated_propensity = calibrated_propensity_regressor.predict_proba(
                                                                X)[:, 1]
    # Threshold propensities
    if propensity_threshold != 0:
        estimated_propensity[estimated_propensity >
                             (1 - propensity_threshold)
                             ] = (1 - propensity_threshold)

        estimated_propensity[estimated_propensity <
                             propensity_threshold
                             ] = propensity_threshold

    # Use a joint estimator for the conditional means
    if RF:
        mu = RandomForestRegressor(n_estimators=n_trees,
                                max_depth=max_depth,
                                n_jobs=n_jobs)
    else:
        mu = LassoCV(cv=cv)

    aggregate = np.concatenate((X, W.reshape(len(W), 1)), axis=1)
    mu.fit(aggregate, Y)

    treated_aggregate = np.hstack((X, np.ones((len(X), 1))))
    treated_potential_outcome = mu.predict(treated_aggregate)
    control_aggregate = np.hstack((X, np.zeros((len(X), 1))))
    control_potential_outcome = mu.predict(control_aggregate)

    inverse_propensity_treated = np.mean((W / estimated_propensity)
                                * (Y - treated_potential_outcome))
    inverse_propensity_control = np.mean((1 - W) / (1 - estimated_propensity)
                                  * (Y - control_potential_outcome))
    regression_term = np.mean(treated_potential_outcome - control_potential_outcome)

    result = (inverse_propensity_treated
              - inverse_propensity_control
              + regression_term)

    return result

