import sys
import os
import time
import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist
from nsd_access import NSDAccess
from utils.nsd_get_data import get_conditions, get_betas
from utils.utils import average_over_conditions

"""
    module to gather the region of interest rdms
"""

# vars
sub = int(sys.argv[1])
sub = f"subj0{sub}"
n_jobs = 1
n_sessions = 5
n_subjects = 3

targetspace = 'fsaverage'

ROIS = {
        1007: "ctx-lh-fusiform", 
        1008: "ctx-lh-inferiorparietal", 
        1009: "ctx-lh-inferiortemporal", 
        1016: "ctx-lh-parahippocampal",
        1025: "ctx-lh-precuneus", 
        1029: "ctx-lh-superiorparietal",
        
        2007: "ctx-rh-fusiform", 
        2008: "ctx-rh-inferiorparietal", 
        2009: "ctx-rh-inferiortemporal", 
        2016: "ctx-rh-parahippocampal",
        2025: "ctx-rh-precuneus", 
        2029: "ctx-rh-superiorparietal",
        }

# set up directories
base_dir = "/work2/07365/sguo19/stampede2/"
nsd_dir = os.path.join(base_dir, 'NSD')
proj_dir = os.path.join(base_dir, 'nsddatapaper_rsa')
betas_dir = os.path.join(proj_dir, 'rsa')

parc_path = os.path.join(nsd_dir, "nsddata", "freesurfer", sub, "mri", "aparc.DKTatlas+aseg.mgz")

# sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
# models_dir = os.path.join(proj_dir, 'rsa', 'serialised_models')

# initiate nsd access
nsda = NSDAccess(nsd_dir)

# path where we save the rdms
outpath = os.path.join(betas_dir, 'roi_analyses')
if not os.path.exists(outpath):
    os.makedirs(outpath)

# load parcellation
maskdata = nib.load(parc_path).get_fdata()

# extract conditions
conditions = get_conditions(nsd_dir, sub, n_sessions)

# we also need to reshape conditions to be ntrials x 1
conditions = np.asarray(conditions).ravel()

# then we find the valid trials for which we do have 3 repetitions.
conditions_bool = [
    True if np.sum(conditions == x) == 3 else False for x in conditions]

conditions_sampled = conditions[conditions_bool]

# find the subject's unique condition list (sample pool)
sample = np.unique(conditions[conditions_bool])

betas_file = os.path.join(
    outpath, f'{sub}_betas_list_{targetspace}.npy'
)
betas_mean_file = os.path.join(
        outpath, f'{sub}_betas_list_{targetspace}_averaged.npy'
)

if not os.path.exists(betas_mean_file):
    # get betas
    betas_mean = get_betas(
        nsd_dir,
        sub,
        n_sessions,
        targetspace=targetspace,
    )
    print(f'concatenating betas for {sub}')
    betas_mean = np.concatenate(betas_mean, axis=1).astype(np.float32)

    print(f'averaging betas for {sub}')
    betas_mean = average_over_conditions(
        betas_mean,
        conditions,
        conditions_sampled,
    ).astype(np.float32)

    # print
    print(f'saving condition averaged betas for {sub}')
    np.save(betas_mean_file, betas_mean)

else:
    print(f'loading betas for {sub}')
    betas_mean = np.load(betas_mean_file, allow_pickle=True)


# print
print(f'saving condition list for {sub}')
np.save(
        os.path.join(
            outpath, f'{sub}_condition_list.npy'
        ),
        conditions_sampled
    )

# save the subject's full ROI RDMs
for roi in range(1, 6):
    mask_name = ROIS[roi]

    rdm_file = os.path.join(
        outpath, f'{sub}_{mask_name}_fullrdm_correlation.npy'
    )

    if not os.path.exists(rdm_file):

        # logical array of mask vertices
        vs_mask = maskdata == roi
        print(f'working on ROI: {mask_name}')

        masked_betas = betas_mean[vs_mask, :]

        good_vox = [
            True if np.sum(
                np.isnan(x)
                ) == 0 else False for x in masked_betas]

        if np.sum(good_vox) != len(good_vox):
            print(f'found some NaN for ROI: {mask_name} - {sub}')

        masked_betas = masked_betas[good_vox, :]

        # prepare for correlation distance
        X = masked_betas.T

        print(f'computing RDM for roi: {mask_name}')
        start_time = time.time()
        rdm = pdist(X, metric='correlation')

        if np.any(np.isnan(rdm)):
            raise ValueError

        elapsed_time = time.time() - start_time
        print(
            'elapsedtime: ',
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
        )
        print(f'saving full rdm for {mask_name} : {sub}')
        np.save(
            rdm_file,
            rdm
        )
