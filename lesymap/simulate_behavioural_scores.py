import numpy as np
import numpy.random as rnd
import pandas as pd

'''
This file contains functions for simulating behavioural scores (outcomes)
for the Lesymap brain lesion dataset.
Each function simulates a different type of lesion/behaviour relationship.
'''


def simulate_behavioural_scores_single(X, roi, amplitude=-1, noise_level=0.5):
    '''
    Simulate behavioural scores in the case where only one ROI affects the
    scores.
    '''
    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)
    behavioural_scores = amplitude * X[roi] + noise_vector

    return behavioural_scores.to_numpy()


def simulate_behavioural_scores_OR(X, rois, amplitude=-1, noise_level=0.5):
    '''
    Simulate behavioural scores in the case where multiple ROIs affect the
    scores, such that behavioural scores are impacted as soon as at least one
    ROI is lesioned, and the effects of the lesioned ROIs are not added
    together. This is equivalent to having an OR operator between the ROIs.
    '''
    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)
    # Making sure rois is a list for pandas indexing
    rois = list(rois)

    concerned_regions = X[rois]
    # Take the max of lesioned voxels which acts like a soft OR operator
    max_lesion = concerned_regions.max(axis=1)
    behavioural_scores = amplitude * max_lesion + noise_vector

    return behavioural_scores.to_numpy()


def simulate_behavioural_scores_AND(X, rois, amplitude=-1, noise_level=0.5):
    '''
    Simulate behavioural scores in the case where multiple ROIs affect the
    scores, such that behavioural scores are impacted only if all of the
    ROIs are lesioned, and the effects of the lesioned ROIs are not added
    together. This is equivalent to having an AND operator between the ROIs.
    '''

    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)
    # Making sure rois is a list for pandas indexing
    rois = list(rois)

    concerned_regions = X[rois]

    # Take the min of lesioned voxels which acts like a soft AND operator
    min_lesion = concerned_regions.min(axis=1)

    behavioural_scores = amplitude * min_lesion + noise_vector

    return behavioural_scores.to_numpy()


def simulate_behavioural_scores_SUM(X, rois, amplitudes, noise_level=0.5):
    '''
    Simulate behavioural scores in the case where multiple ROIs affect the
    scores, such that behavioural scores are impacted as soon as at least one
    ROI is lesioned, and the effects of the lesioned ROIs are added together.
    This is equivalent to having a "+" operator between the ROIs.
    '''
    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)

    rois = list(rois)

    behavioural_scores = noise_vector
    for i, roi in enumerate(rois):
        behavioural_scores += amplitudes[i] * X[roi]

    return behavioural_scores.to_numpy()


def simulate_behavioural_scores_XOR(X, rois, amplitude=-1, noise_level=0.5):
    '''
    Simulate behavioural scores in the case where two ROIs affect the
    scores, such that behavioural scores are impacted only if exactly one of the
    ROIs are lesioned. This is equivalent to having an XOR operator between ROIs.
    '''

    assert(len(rois) == 2)
    
    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)
    # Making sure rois is a list for pandas indexing
    rois = list(rois)

    concerned_regions = X[rois]
    
    # Take the max-min of lesioned voxels which acts like a soft XOR operator
    max_min_lesion = concerned_regions.max(axis=1)-concerned_regions.min(axis=1)
    print("nombre de 1 :", max_min_lesion.sum())
    
    behavioural_scores = amplitude * max_min_lesion + noise_vector

    return behavioural_scores.to_numpy()


def simulate_behavioural_scores_ANDORAND(X, rois, amplitude=-1, noise_level=0.5):
    '''
    Simulate behavioural scores in the case where four ROIs affect the
    scores, such that behavioural scores are impacted only if the first two ROIs     or the last two ROIS are lesioned simultaneously.
    '''
    
    assert(len(rois) == 4)
    
    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)
    # Making sure rois is a list for pandas indexing
    rois = list(rois)
    
    rois1 = [rois[0], rois[1]]
    concerned_regions1 = X[rois1]
    min_lesion1 = concerned_regions1.min(axis=1)
    rois2 = [rois[2], rois[3]]
    concerned_regions2 = X[rois2]
    min_lesion2 = concerned_regions2.min(axis=1)
    
    les = pd.concat([min_lesion1, min_lesion2], axis=1)
    
    andorand_lesion = les.max(axis=1)
    
    behavioural_scores = amplitude * andorand_lesion + noise_vector

    return behavioural_scores.to_numpy()
    
    
def simulate_behavioural_scores_ORANDOR(X, rois, amplitude=-1, noise_level=0.5):
    '''
    Simulate behavioural scores in the case where two ROIs affect the
    scores, such that behavioural scores are impacted only if one of the first       two ROIs and one of the last two ROIS are lesioned simultaneously.
    '''

    
    assert(len(rois) == 4)
    
    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)
    # Making sure rois is a list for pandas indexing
    rois = list(rois)
    
    rois1 = [rois[0], rois[1]]
    concerned_regions1 = X[rois1]
    max_lesion1 = concerned_regions1.max(axis=1)
    rois2 = [rois[2], rois[3]]
    concerned_regions2 = X[rois2]
    max_lesion2 = concerned_regions2.max(axis=1)
    
    les = pd.concat([max_lesion1, max_lesion2], axis=1)
    
    orandor_lesion = les.min(axis=1)
    
    behavioural_scores = amplitude * orandor_lesion + noise_vector

    return behavioural_scores.to_numpy()