import argparse
# import sys
import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io
from scipy.spatial.distance import pdist
from nsd_access import NSDAccess
from utils.nsd_get_data import get_conditions, get_betas
from utils.utils import average_over_conditions
from config import *

"""
    module to gather the region of interest rdms
"""

# parser = argparse.ArgumentParser()
# parser.add_argument("sub", help="subject id in integer. e.g., '1' for subj01", type=int, default=1)
# parser.add_argument("n_sessions", help="n_sessions to load", type=int, default=10)
# # parser.add_argument("n_jobs", help="n_jobs to run", type=int, default=1)
# args = parser.parse_args()

# sub = f"subj0{args.sub}"
# n_sessions = args.n_sessions
# # n_jobs = args.n_jobs

# ===== vars & paths
# sub = int(sys.argv[1])
# sub = f"subj0{sub}"
# n_jobs = 1
# n_sessions = 10

targetspace = 'fsaverage'

# # DKT atlas ROIS
# ROIS = {
#         1007: "ctx-lh-fusiform", 
#         1008: "ctx-lh-inferiorparietal", 
#         1009: "ctx-lh-inferiortemporal", 
#         1016: "ctx-lh-parahippocampal",
#         1025: "ctx-lh-precuneus", 
#         1029: "ctx-lh-superiorparietal",
        
#         2007: "ctx-rh-fusiform", 
#         2008: "ctx-rh-inferiorparietal", 
#         2009: "ctx-rh-inferiortemporal", 
#         2016: "ctx-rh-parahippocampal",
#         2025: "ctx-rh-precuneus", 
#         2029: "ctx-rh-superiorparietal",
#         }

# # glasser atlas
# ROIS = ["parietal"]  #"ventral"]    #"SPL", "IPL",] # "PCV"]
# ROIS = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']
# group_level = "pathway"
# ROIS = ["dorsal"]

# set up directories
base_dir = "/work2/07365/sguo19/stampede2/"
nsd_dir = os.path.join(base_dir, 'NSD')
proj_dir = os.path.join(base_dir, 'nsddatapaper_rsa')
betas_dir = os.path.join(proj_dir, 'rsa')

# sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
# models_dir = os.path.join(proj_dir, 'rsa', 'serialised_models')

# initiate nsd access
nsda = NSDAccess(nsd_dir)

# path where we save the rdms
outpath = os.path.join(betas_dir, 'roi_analyses')
if not os.path.exists(outpath):
    os.makedirs(outpath)

# ===== loading
# === load parcellation
# parc_path = os.path.join(nsd_dir, "nsddata", "freesurfer", sub, "mri", "aparc.DKTatlas+aseg.mgz")
if "pVTC" in ROIS:
    lh_file = os.path.join('./lh.highlevelvisual.mat')
    rh_file = os.path.join('./rh.highlevelvisual.mat')
    maskdata_lh = scipy.io.loadmat(lh_file)["lh"].squeeze()
    maskdata_rh = scipy.io.loadmat(rh_file)["rh"].squeeze()
else:
    lh_file = os.path.join(nsd_dir, "nsddata", "freesurfer", "fsaverage", "label", 'lh.HCP_MMP1.mgz')
    rh_file = os.path.join(nsd_dir, "nsddata", "freesurfer", "fsaverage", "label", 'rh.HCP_MMP1.mgz')
    # maskdata = nib.load(parc_path).get_fdata().flatten()
    maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
    maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

maskdata = np.hstack((maskdata_lh, maskdata_rh))
print("loaded maskdata: ", maskdata.shape)

# === load ROIs
ROI_path = os.path.join(proj_dir, "utils", "ROI_labels.tsv")
ROI_df = pd.read_csv(ROI_path, sep='\t')

# === load conditions
conditions = get_conditions(nsd_dir, sub, n_sessions)
conditions = np.asarray(conditions).ravel()  # ntrials x 1

# then we find the valid trials for which we do have 3 repetitions.
conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
conditions_sampled = conditions[conditions_bool]
# find the subject's unique condition list (sample pool)
sample = np.unique(conditions[conditions_bool])
print("sample pool size:", len(sample))

# save conditions
cond_out_path = os.path.join(outpath, f'{sub}_condition-list_session-{n_sessions}.npy')
print(f'saving condition list for {sub}')
np.save(cond_out_path, conditions_sampled)
print("condition shape:", conditions_sampled.shape)

# === load mean betas / calculate from full betas
betas_mean_file = os.path.join(
        outpath, f'{sub}_betas-list_{targetspace}_session-{n_sessions}_averaged.npy'
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
print("betas_mean: ", betas_mean.shape)

# ===== make ROI RDMs & save
for mask_name in ROIS:

    rdm_file = os.path.join(
        outpath, f'{sub}_{mask_name}_fullrdm_correlation_session-{n_sessions}.npy'
    )

    if not os.path.exists(rdm_file):
        print(f'working on ROI: {mask_name}')

        # depending on grouping level, limit mask_df to corresponding small regions
        mask_df = ROI_df[ ROI_df[group_level] == mask_name ]

        # make mask
        mask = np.array([False for _ in range(len(maskdata))])
        for roi, roi_name in zip(mask_df["ROI_id"], mask_df["ROI_name"]):
            mask = mask | (maskdata == roi)
        print("mask shape:", mask.shape)

        masked_betas = betas_mean[mask, :]
        print("masked_betas shape:", masked_betas.shape)

        good_vox = [
            True if np.sum(
                np.isnan(x)
                ) == 0 else False for x in masked_betas]

        if np.sum(good_vox) != len(good_vox):
            print(f'found some NaN for ROI: {mask_name} - {sub}')

        masked_betas = masked_betas[good_vox, :]
        print("masked_betas shape after filtering with good_vox", masked_betas.shape)

        # prepare for correlation distance
        X = masked_betas.T
        print("X shape:", X.shape)

        print(f'computing RDM for roi: {mask_name}')
        start_time = time.time()
        rdm = pdist(X, metric='correlation')
        print("rdm shape", rdm.shape)

        if np.any(np.isnan(rdm)):
            raise ValueError("found nan in rdm")

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
