# import sys
import argparse
import os
import time
import numpy as np
# import nibabel as nib
import scipy.io
from scipy.spatial.distance import pdist
from nsd_access import NSDAccess
from utils.nsd_get_data import get_conditions, get_betas
from utils.utils import average_over_conditions

"""
    module to gather the region of interest rdms
"""
# sub = int(sys.argv[1])
# n_jobs = 6
n_sessions = 20
n_subjects = 3
subs = ['subj0{}'.format(x+6) for x in range(n_subjects)]

# parser = argparse.ArgumentParser()
# parser.add_argument("sub", help="subject id in integer. e.g., '1' for subj01", type=int, default=1)
# parser.add_argument("n_sessions", help="n_sessions to load", type=int, default=10)
# parser.add_argument("n_jobs", help="n_jobs to run", type=int, default=1)
# args = parser.parse_args()

# sub = f"subj0{args.sub}"
# n_sessions = args.n_sessions
# n_jobs = args.n_jobs

# set up directories
base_dir = "/work2/07365/sguo19/stampede2/"
nsd_dir = os.path.join(base_dir, 'NSD')
proj_dir = os.path.join(base_dir, 'nsddatapaper_rsa')
betas_dir = os.path.join(proj_dir, 'rsa')

# initiate nsd access
nsda = NSDAccess(nsd_dir)

# path where we save the rdms
outpath = os.path.join(betas_dir, 'roi_analyses')
if not os.path.exists(outpath):
    os.makedirs(outpath)

# we use the fsaverage space.
targetspace = 'fsaverage'

# lh_file = os.path.join(proj_dir, 'lh.highlevelvisual.mgz')
# rh_file = os.path.join(proj_dir, 'rh.highlevelvisual.mgz')
# # load the lh mask
# maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
# maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

lh_file = os.path.join('./lh.highlevelvisual.mat')
rh_file = os.path.join('./rh.highlevelvisual.mat')
maskdata_lh = scipy.io.loadmat(lh_file)["lh"].squeeze()
maskdata_rh = scipy.io.loadmat(rh_file)["rh"].squeeze()

maskdata = np.hstack((maskdata_lh, maskdata_rh))

ROIS = {1: 'pVTC', 2: 'aVTC', 3: 'v1', 4: 'v2', 5: 'v3'}

roi_names = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']

# # sessions
# n_sessions = 40


for sub in subs: 
    print(f"\n***** Processing {sub}... *****\n")

    # extract conditions
    conditions = get_conditions(nsd_dir, sub, n_sessions)

    # we also need to reshape conditions to be ntrials x 1
    conditions = np.asarray(conditions).ravel()

    # then we find the valid trials for which we do have 3 repetitions.
    conditions_bool = [
        True if np.sum(conditions == x) == 3 else False for x in conditions]

    conditions_sampled = conditions[conditions_bool]

    # # find the subject's unique condition list (sample pool)
    # sample = np.unique(conditions[conditions_bool])

    betas_file = os.path.join(
        outpath, f'{sub}_betas_list_{targetspace}.npy'
    )
    betas_mean_file = os.path.join(
            outpath, f'{sub}_betas_list_{targetspace}_session-{n_sessions}_averaged.npy'
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
                outpath, f'{sub}_condition_list_session-{n_sessions}.npy'
            ),
            conditions_sampled
        )

    # save the subject's full ROI RDMs
    for roi in range(1, 6):
        mask_name = ROIS[roi]

        rdm_file = os.path.join(
            outpath, f'{sub}_{mask_name}_fullrdm_correlation_session-{n_sessions}.npy'
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