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
    Simulate behavioural scores in the case where multiple ROIs affect the
    scores, such that behavioural scores are impacted only if exactly one of the
    ROIs are lesioned. This is equivalent to having an XOR operator between ROIs.
    '''

    n_subjects = X.shape[0]
    noise_vector = rnd.normal(size=n_subjects, scale=noise_level)
    # Making sure rois is a list for pandas indexing
    rois = list(rois)

    concerned_regions = X[rois]

    # Take the max-min of lesioned voxels which acts like a soft XOR operator
    max_min_lesion = concerned_regions.max(axis=1) - concerned_regions.min(axis=1)

    behavioural_scores = amplitude * max_min_lesion + noise_vector

    return behavioural_scores.to_numpy()
