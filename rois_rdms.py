"""
This script loads betas for the filtered trials, computes the average of betas for each image, 
filters for good voxels, then computes the RDM for the betas.

Output:
"condition-list"/conditions_sampled: save filtered trialIDs with 3 repetitions;
"betas-list"/betas_mean: save average of betas for each image presented;
"fullrdm"/rdm: RDM of average betas for all images for each ROI for each subject.
"""
import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io
from scipy.spatial.distance import pdist
from nsd_access import NSDAccess
from utils.nsd_get_data import get_conditions, get_betas, get_labels
from utils.utils import average_over_conditions
from config import *

def rois_rdms(sub = "subj01",
                n_sessions = 20,
                n_jobs=1,
                group_level="ROI_name",
                ROIS = ['pVTC', 'aVTC', 'v1', 'v2', 'v3'],
                rdm_dim = "trial",
                ):
    """
    Input: 
    group_level: the level to group ROIs by as in pd.groupby from ROI_labels.tsv;
    ROIS: list of ROIs to compute RDM for. Must be found under group_level in ROI_labels.tsv;
    rdm_dim: possible values: ["trial", "cate"]. Dimension to do rdm on. 
             Currently computes "voxel_category_betas" with matrix multiply. 
             TODO: when "cate", average betas across categories rather than trials.
    See top of the script for output/saved files.
    """

    targetspace = 'fsaverage'

    # set up directories
    base_dir = "/work2/07365/sguo19/stampede2/"
    nsd_dir = os.path.join(base_dir, 'NSD')
    proj_dir = os.path.join(base_dir, 'nsddatapaper_rsa')
    betas_dir = os.path.join(proj_dir, 'rsa')

    # path where we save the rdms
    outpath = os.path.join(betas_dir, 'roi_analyses', sub)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # ===== loading
    # === load parcellation
    if "pVTC" in ROIS:
        # ventral stream masks from original NSD paper RSA analyses. 
        # ROI values included in ROI_labels.tsv.
        # .mat files are generated from .mgz files found here: 
        # https://github.com/cvnlab/nsddatapaper/tree/main/mainfigures/SCIENCE.RSA
        lh_file = os.path.join('./lh.highlevelvisual.mat')
        rh_file = os.path.join('./rh.highlevelvisual.mat')
        maskdata_lh = scipy.io.loadmat(lh_file)["lh"].squeeze()
        maskdata_rh = scipy.io.loadmat(rh_file)["rh"].squeeze()
    else:
        # dorsal stream regions defined by HCP_MMP1/Glasser atlas
        lh_file = os.path.join(nsd_dir, "nsddata", "freesurfer", "fsaverage", "label", 'lh.HCP_MMP1.mgz')
        rh_file = os.path.join(nsd_dir, "nsddata", "freesurfer", "fsaverage", "label", 'rh.HCP_MMP1.mgz')
        maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
        maskdata_rh = nib.load(rh_file).get_fdata().squeeze()

    maskdata = np.hstack((maskdata_lh, maskdata_rh))
    print("loaded maskdata: ", maskdata.shape)

    # === load ROI labels
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

    # save *non-unique* conditions
    cond_out_path = os.path.join(outpath, f'{sub}_condition-list_session-{n_sessions}.npy')
    print(f'saving condition list for {sub}')
    np.save(cond_out_path, conditions_sampled)
    print("condition shape:", conditions_sampled.shape)

    # === load mean betas / calculate from full betas
    betas_mean_file = os.path.join(
            outpath, f'{sub}_betas-list_{targetspace}_session-{n_sessions}_averaged.npy'
    )

    if not os.path.exists(betas_mean_file):
        print(f"generating average betas for {sub}")
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
        print(f'loading saved betas for {sub}')
        betas_mean = np.load(betas_mean_file, allow_pickle=True)
    print("betas_mean: ", betas_mean.shape)

    # ===== make ROI RDMs & save
    for mask_name in ROIS:
        if rdm_dim == "trial": 
            rdm_file = os.path.join(
                outpath, f'{sub}_{mask_name}_fullrdm_correlation_session-{n_sessions}.npy'
            )
        else: 
            rdm_file = os.path.join(
                outpath, f'{sub}_{mask_name}_fullrdm-{rdm_dim}_correlation_session-{n_sessions}.npy'
            )
            
        if not os.path.exists(rdm_file):
            print(f'working on ROI: {mask_name}')

            # depending on grouping level, limit mask_df to corresponding small regions
            mask_df = ROI_df[ ROI_df[group_level] == mask_name ]

            # === make mask
            mask = np.zeros(len(maskdata)).astype(bool)
            for roi in mask_df["ROI_id"]:
                mask = mask | (maskdata == roi)  # OR all the voxels together to one mask
            print("mask shape:", mask.shape)

            masked_betas = betas_mean[mask, :]
            print("masked_betas shape:", masked_betas.shape)

            good_vox = [
                True if np.sum(
                    np.isnan(x)
                    ) == 0 else False for x in masked_betas]

            # n_selected_voxels x n_trials
            if np.sum(good_vox) != len(good_vox):
                print(f'found some NaN for ROI: {mask_name} - {sub}')

            masked_betas = masked_betas[good_vox, :]
            print("masked_betas shape after filtering with good_vox", masked_betas.shape)

            # === prepare for correlation distance
            # TODO: move this to above betas averaging & add average across cate 
            if rdm_dim == "trial": 
                # trial tSNE: dots are indiv image shown
                X = masked_betas.T
            elif rdm_dim == "cate":
                # category tSNE: 80 dots for COCO cates
                # *** the underlying nsd_access method is modified with Parallel. Original does not take n_jobs ***
                label_matrix = get_labels(sub, betas_dir, nsd_dir, sample-1, n_sessions=n_sessions, n_jobs=n_jobs)
                # make vox-by-cate mat
                print(f"making voxel category matrix...")
                # matmal to get sum of product
                vox_cate_mat = np.dot(masked_betas, label_matrix)
                # normalize by cate frequency
                freq = label_matrix.sum(axis=0)
                # vox_cate_mat: voxel x 80
                vox_cate_mat = np.divide(vox_cate_mat, freq, out=np.zeros_like(vox_cate_mat), where=freq!=0)
                X = vox_cate_mat.T

            print(f"X shape with rdm dim {rdm_dim}:", X.shape)  # n_features x n_voxels

            # === compute RDM
            print(f'computing RDM for roi: {mask_name}')
            start_time = time.time()
            rdm = pdist(X, metric='correlation')
            print("rdm shape", rdm.shape)

            if np.any(np.isnan(rdm)):
                # raise ValueError("found nan in rdm")
                print("found nan in rdm. abort.")
                return 

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


if __name__ == "__main__":
    from config import *

    n_subjects = 8
    subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]  
    ROI_dicts = {  
        # group_level: ROIS
        "pathway": ["ventral"],
        "region": ["aVTC", "pVTC", "v1", "v2", "v3"],
        "pathway": ["dorsal"],
        "region": ["SPL", "IPL", "PCC"],
    }

    for sub in subs:  
        print(f"\n***** Running rois_rdms on {sub}... *****")
        for group_level, ROIS in ROI_dicts.items():
            print(f"*** running {group_level} with {ROIS} ***")

            rois_rdms(sub=sub,
                        n_sessions=n_sessions,
                        n_jobs=n_jobs,
                        group_level=group_level,
                        ROIS=ROIS,
                        rdm_dim = rdm_dim,
                        )