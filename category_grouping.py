"""
This script groups 80 COCO categories with cortical ROI activities and make plots of 
these categories in low dimensional spaces, 
i.e., similar operation as before for unique images, but now for categories.
(See presentation slides for details)
"voxel_category_matrix" from dot(betas, image_category_matrix).
Dimensionality reduction with PCA.
"""

import os
import numpy as np
from nsd_access import NSDAccess
from sklearn.decomposition import PCA
import nibabel as nib
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.nsd_get_data import get_conditions, get_labels
from config import *

def category_grouping(sub = "subj01",
                n_sessions = 20,
                n_jobs=1,
                group_level="pathway",
                ROIS = ["dorsal"],
                plot_3D = False,  # takes a long time to make gif so only do if needed
                ):
    # ===== setup directories
    base_dir = "/work2/07365/sguo19/stampede2/"
    nsd_dir = os.path.join(base_dir, 'NSD')
    proj_dir = os.path.join(base_dir, 'nsddatapaper_rsa')

    betas_dir = os.path.join(proj_dir, 'rsa')
    figures_outpath = os.path.join(betas_dir, 'roi_analyses')
    outpath = os.path.join(betas_dir, 'roi_analyses', sub)

    # ========== same loading process as in roi_rdm.py ===========
    # ===== load labels
    # all categories in COCO (80)
    all_labels = np.array(pd.read_csv("./utils/COCO_labels.tsv", sep="\t")["label"])
    # load category label
    label_file = os.path.join(betas_dir, f'{sub}_sample_labels_session-{n_sessions}.npy')

    print(f"loading label matrix for {sub} with {n_sessions}...")
    if os.path.exists(label_file):
        label_matrix = np.load(label_file, allow_pickle=True)
    else:
        # load conditions data
        conditions = get_conditions(nsd_dir, sub, n_sessions)
        conditions = np.asarray(conditions).ravel()  # ntrials x 1
        # then we find the valid trials for which we do have 3 repetitions.
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions]
        # find the subject's condition list (sample pool)
        sample = np.unique(conditions[conditions_bool])

        # *** the underlying nsd_access method is modified with Parallel. Original does not take n_jobs ***
        label_matrix = get_labels(sub, betas_dir, nsd_dir, sample-1, n_sessions=n_sessions, n_jobs=n_jobs)

    print(f"labels shape: {all_labels.shape}, {label_matrix.shape}")

    # ===== load masks
    print(f"Loading betas for {sub}...")
    lh_file = os.path.join(nsd_dir, "nsddata", "freesurfer", "fsaverage", "label", 'lh.HCP_MMP1.mgz')
    rh_file = os.path.join(nsd_dir, "nsddata", "freesurfer", "fsaverage", "label", 'rh.HCP_MMP1.mgz')
    maskdata_lh = nib.load(lh_file).get_fdata().squeeze()
    maskdata_rh = nib.load(rh_file).get_fdata().squeeze()
    maskdata = np.hstack((maskdata_lh, maskdata_rh))

    ROI_path = os.path.join(proj_dir, "utils", "ROI_labels.tsv")
    ROI_df = pd.read_csv(ROI_path, sep='\t')

    # ===== load betas
    print(f"Loading betas for {sub} with {n_sessions}...")
    betas_mean = np.load(os.path.join(outpath, f"{sub}_betas-list_fsaverage_session-{n_sessions}_averaged.npy"),
                allow_pickle=True)
    print(f"betas shape: {betas_mean.shape}")

    # for each ROI
    for mask_name in ROIS:
        # === make mask
        # depending on grouping level, limit mask_df to corresponding small regions
        mask_df = ROI_df[ ROI_df[group_level] == mask_name ]

        mask = np.array([False for _ in range(len(maskdata))])
        for roi, _ in zip(mask_df["ROI_id"], mask_df["ROI_name"]):
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

        # ========== new stuff ==========
        # === make voxel by category matrix
        print(f"making voxel category matrix...")
        # matmal to get sum of product
        vox_cate_mat = np.dot(masked_betas, label_matrix)
        # normalize by cate frequency
        freq = label_matrix.sum(axis=0)
        # vox_cate_mat: voxel x 80
        vox_cate_mat = np.divide(vox_cate_mat, freq, out=np.zeros_like(vox_cate_mat), where=freq!=0)

        print(f"Doing PCA...")
        pca = PCA(n_components=n_pcs)
        pca = pca.fit(vox_cate_mat)
        pcs = pca.components_
        np.save(f"{figures_outpath}/cate_pcs/{sub}_{mask_name}_{n_sessions}_pca.npy", pca)
        print(f"saved {mask_name} pca object for {sub}!")

        # make normalized cate-color map & color categories by the PC value they have 
        colors = np.zeros((80,4))
        for i in range(3):
            colors[:,i] = (pcs[i] - pcs[i].min()) / (pcs[i].max() - pcs[i].min()).T
        colors[:,3] += 1
        # save colormap
        np.save(f"{figures_outpath}/cate_pcs/{sub}_{mask_name}_{n_sessions}_colormap.npy", colors)
        print(f"Color axis saved!")

        # ===== plot categories in PC space
        # === 2D plots
        print("plotting 2D plots...")
        fig, axes = plt.subplots(3,1,figsize=(10,24))

        ax = axes[0]
        ax.scatter(pcs[0], pcs[1], c=colors)
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        for i in range(80): 
            ax.text(pcs[0,i],pcs[1,i], all_labels[i], size=15, zorder=1,  color=colors[i], alpha=0.7) 

        ax = axes[1]
        ax.scatter(pcs[0], pcs[2], c=colors)
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc3')
        for i in range(80): 
            ax.text(pcs[0,i],pcs[2,i], all_labels[i], size=15, zorder=1,  color=colors[i], alpha=0.7) 

        ax = axes[2]
        ax.scatter(pcs[1], pcs[2], c=colors)
        ax.set_xlabel('pc2')
        ax.set_ylabel('pc3')
        for i in range(80): 
            ax.text(pcs[1,i],pcs[2,i], all_labels[i], size=15, zorder=1,  color=colors[i], alpha=0.7) 
            
        plt.tight_layout()
        plt.savefig(f"{figures_outpath}/cate_pcs/{sub}_{mask_name}_{n_sessions}_category_pcs.png", dpi=300)
        plt.close('all')
        print("saved 2D category PC plots!")

        # === 3D plots
        if plot_3D: 
            print("plotting 3D plots...")
            fig = plt.figure(figsize=(15,15))
            ax = plt.axes(projection='3d')
            ax.scatter(pcs[0], pcs[1], pcs[2], s=50, c=colors)
            ax.set_xlabel('pc1')
            ax.set_ylabel('pc2')
            ax.set_zlabel('pc3')

            for i in range(len(pcs[0])): #plot each point + it's index as text above
                ax.text(pcs[0,i],pcs[1,i],pcs[2,i], all_labels[i], size=15, zorder=1, color=colors[i], alpha=0.5) 
                
            # make gif of 3d plot
            def rotate(angle):
                ax.view_init(azim=angle)
            rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=120)
            rot_animation.save(f'{figures_outpath}/cate_pcs/{sub}_{mask_name}_{n_sessions}_category_pcs_rotation.gif', 
                            dpi=100, writer='imagemagick')
            plt.close('all')
            print("saved 3D category PC plot!")


if __name__ == "__main__":
    from config import *

    n_subjects = 8
    subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]  
    ROI_dicts = {
        "pathway": ["dorsal"],
        # "pathway": ["ventral"],
    }

    for sub in subs:  
        print(f"\n***** Plotting tSNE for {sub}... *****")
        for group_level, ROIS in ROI_dicts.items():
            print(f"*** running {group_level} with {ROIS} ***")

            category_grouping(sub=sub,
                        n_sessions=n_sessions,  #from config
                        n_jobs=n_jobs,          #from config
                        group_level=group_level,
                        ROIS=ROIS,
                        plot_3D=plot_3D,        #from config
                        )

