# -*- encoding: utf-8 -*-
'''Utility functions for the pre-processing of LESYMAP data.'''

import numpy as np
import pandas as pd
import nilearn
from joblib import Parallel, delayed
from nilearn.input_data import NiftiMasker


def find_lesioned_regions(lesion_map, atlas_img=None):
    '''
    Finds the lesioned regions of an lesion map, according to a provided atlas.

    Inputs
    ------
    lesion_map : NIFTI image
        The nifti image representing the brain lesions.
        Lesioned voxels should have a value equal to 1, and healthy
        voxels a value equal to 0.
    atlas_img : NIFTI image
        Brain atlas to use for parcellation.
        Default is the atlas provided in the LESYMAP dataset.

    Outputs
    -------
    lesioned_regions : list of length (n_regions)
        Contains the proportion of lesioned voxels in each atlas region.
    '''

    # If no atlas image is provided, load the default one from the
    # LESYMAP dataset
    if atlas_img is None:

        path_to_parcellation = "/storage/tompouce/lucmarti/lesymap/extdata/template/Parcellation_403areas.nii.gz"
        atlas_img = nilearn.image.load_img(path_to_parcellation)

    # Mask atlas image and lesion image

    nifti_masker = NiftiMasker(mask_strategy='template')
    lesion_array = nifti_masker.fit_transform(lesion_map)
    atlas_array = nifti_masker.fit_transform(atlas_img)
    atlas_array = atlas_array.astype(int)

    # Find lesioned voxels in the original image
    lesioned_regions = []
    for region in np.unique(atlas_array):

        # Find voxels belonging to the region
        region_voxels = np.where(atlas_array == region)

        # Compute number of voxels in region
        region_size = len(region_voxels[0])
        # Compute number of voxels with lesion in the region
        number_of_lesioned_voxels_in_region = np.sum(lesion_array[region_voxels])

        # Add proportion of lesioned voxels
        lesioned_regions.append(number_of_lesioned_voxels_in_region
                                / region_size)
    return lesioned_regions


def _dataframe_lesioned_regions(p, column_names, atlas_img=None):
    '''
    Joblib wrapper function which puts lesion status of regions into dataframe
    '''
    lesion_map = nilearn.image.load_img(p)
    lesioned_regions = find_lesioned_regions(lesion_map, atlas_img)

    region_status_df = pd.DataFrame([lesioned_regions], columns=column_names)

    return region_status_df


def build_lesion_dataset(paths_to_lesion_files=None, column_names=None,
                         atlas_img=None, n_jobs=10):
    '''
    Builds a dataset where each feature is the proportion of lesioned voxels
    in the corresponding atlas region

    Inputs
    ------
    paths_to_lesion_files : list of str
        Filepaths to the lesion maps. Default are the 131 lesion maps
        of the LESYMAP dataset, in my personal folder on DRAGO.
    column_names : list of str
        Names to give to each atlas region.
        Default are the region names of the atlas provided in the
        LESYMAP dataset.
    atlas_img : NIFTI image
        Brain atlas to use for parcellation.
        Default is the atlas provided in the LESYMAP dataset.
    n_jobs : int
        Number of threads to use for parallel computing.
        Passed to joblib.Parallel

    Outputs
    -------
    df : Pandas DataFrame of shape (n_subjects, n_regions)
        Dataframe containing the proportion of lesioned voxels
        in the atlas regions.
        Columns correspond to atlas regions, and rows to subjects.
    '''

    # Default lesion file path
    if paths_to_lesion_files is None:
        subject_index = [str(i).zfill(3) for i in range(1, 132)]
        paths_to_lesion_files = ["/storage/tompouce/lucmarti/lesymap/extdata/lesions/Subject_{}.nii.gz".format(s)
                                 for s in subject_index]

    # Each column corresponds to an atlas region,
    # hardcoded column names for now
    # This looks terrible, but the alternative would
    # be passing the atlas array as argument, which
    # would eat up a lot of memory unnecessarily.
    if column_names is None:
        column_names = (list(range(181)) + list(range(201, 381)) +
                        list(range(1219, 1247)) + list(range(2001, 2005))
                        + list(range(2009, 2019)) + [2020])

    df = pd.DataFrame(columns=column_names)

    region_status_vectors = (Parallel(n_jobs=n_jobs, verbose=10)
                             (delayed(_dataframe_lesioned_regions)(p,
                                                                   column_names,
                                                                   atlas_img)
                              for p in paths_to_lesion_files))

    df = df.append(region_status_vectors, ignore_index=True)
    return df


def cull_dataset(X, threshold=True, lesion_threshold=0.6):
    ''' Culls the regions without stroke from a dataset.
        Optionally thresholds the lesion status of each region"

    Inputs
    ------
    X : Pandas DataFrame of shape (n_subjects, n_regions)
        Contains the lesion status of the different
        brain regions from the lesion map
    threshold : bool
        Set to True to threshold the lesion status of each region.
    lesion_threshold : float
        The proportion of lesioned voxels above which a region is
        considered to be lesioned.
        Not taken into account if threshold is set to False.

    Outputs
    -------
    X_culled : Pandas DataFrame
    The original dataframe where the columns corresponding to
    regions that weren't lesioned in any subjects
    are removed.
    '''
    if threshold:
        X[X > lesion_threshold] = 1
        X[X <= lesion_threshold] = 0

    useful_regions = (X != 0).any()
    X_culled = X.loc[:, useful_regions]
    return X_culled
