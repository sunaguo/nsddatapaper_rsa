import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scprep
# import sys
import time
# import tkinter # noqa
from matplotlib import cm
from nsd_access import NSDAccess
from utils.nsd_get_data import get_conditions, get_labels
from scipy.spatial.distance import squareform
from sklearn import manifold
from utils.utils import category_dict, mds

from config import *

"""[nsd_plot_tsne]

    module to plot TSNE or MDS for RDMs computed from
    betas along the visual ventral stream

    example use:

    python nsd_plot_tsne.py 0 1
"""

# parser = argparse.ArgumentParser()
# parser.add_argument("sub", help="subject id in integer. e.g., '1' for subj01", type=int, default=1)
# parser.add_argument("n_sessions", help="n_sessions to load", type=int, default=10)
# parser.add_argument("n_jobs", help="n_jobs to run", type=int, default=1)
# args = parser.parse_args()

# sub = f"subj0{args.sub}"
# n_sessions = args.n_sessions
# n_jobs = args.n_jobs

# subject = int(sys.argv[1])
# sub = f"subj0{subject}"
# # n_jobs = int(sys.argv[2])
# n_jobs = 8

# n_sessions = 10
# n_subjects = 1

# here are the Regions of interest and their indices
# ROIS = {
#     1: 'pVTC',
#     2: 'aVTC',
#     3: 'v1',
#     4: 'v2',
#     5: 'v3'
# }
# ROIS = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']
# ROIS = ["SPL", "IPL",] # "PCV"]  "parietal"]
# group_level = "pathway"
# ROIS = ["dorsal"]

# ===== set up directories
base_dir = "/work2/07365/sguo19/stampede2/"
nsd_dir = os.path.join(base_dir, 'NSD')
proj_dir = os.path.join(base_dir, 'nsddatapaper_rsa')
betas_dir = os.path.join(proj_dir, 'rsa')
# sem_dir = os.path.join(proj_dir, 'derivatives', 'ecoset')
# models_dir = os.path.join(proj_dir, 'rsa', 'serialised_models')

outpath = os.path.join(betas_dir, 'roi_analyses')
if not os.path.exists(outpath):
    os.makedirs(outpath)

tsne_figures = os.path.join(
        outpath, "self_tsne_figures"  #'tsne_figures'
)

if not os.path.exists(tsne_figures):
    os.makedirs(tsne_figures)

# for mask_name in ROIS:
#     category_figures = os.path.join(
#             outpath, 'category_figures', sub, mask_name
#     )

#     if not os.path.exists(category_figures):
#         os.makedirs(category_figures)

nsda = NSDAccess(nsd_dir)

# load conditions data
conditions = get_conditions(nsd_dir, sub, n_sessions)
conditions = np.asarray(conditions).ravel()  # ntrials x 1
# then we find the valid trials for which we do have 3 repetitions.
conditions_bool = [
    True if np.sum(conditions == x) == 3 else False for x in conditions]
# find the subject's condition list (sample pool)
sample = np.unique(conditions[conditions_bool])

# retrieve the category matrix for the sample
category_matrix = get_labels(sub, betas_dir, nsd_dir, sample-1, n_sessions=n_sessions, n_jobs=n_jobs)

# # ===== labels for binary maps
# # TODO: order categories by word embedding rather than alphabetical
# # load labels
# labels = np.load(
#     os.path.join(
#         betas_dir,
#         f'all_stims_category_labels_session-{n_sessions}.npy'
#     ),
#     allow_pickle=True
# )

# # restrain to NSD images 
# all_labels = sorted(list(set(np.concatenate(labels))))

# # also prepare the category binary maps
# category_classes = []
# for cat_i in range(80):
#     flat = np.full(len(sample), '0_unknown')
#     flat[category_matrix[:, cat_i] == 1] = all_labels[cat_i]
#     category_classes.append(flat)

# # get unique colour per category
# category_colors = cm.RdYlBu(range(80))  # there are only 80 categories in COCO in general??

# colormap
cate_colors = ["red", "gold", "grey", "mediumorchid", "limegreen", "cyan", "blue", ]

# prepare the class labels
class_labels = []
for categ_v in category_matrix:

    # category_dict: 1 is animate, 0 inanimate
    cat_is = np.where(categ_v)[0]
    anim_class = [category_dict[str(x)] for x in cat_is]

    n_anim = np.sum(anim_class)

    # special case only people
    if len(cat_is) == 1 and cat_is == 49:
        class_label = 'a_people'

    # people with other animates
    elif 49 in cat_is and n_anim == len(anim_class):
        class_label = 'a_people_animates'

    # people with inanimates
    elif 49 in cat_is and n_anim == 1:
        class_label = 'a_people_inanimates'

    # people with both animates and inanimates
    elif 49 in cat_is and n_anim < len(anim_class):
        class_label = 'a_people_animates_inanimates'

    # all ones? only animate
    elif n_anim == len(anim_class):
        class_label = 'animates'

    elif np.sum(anim_class) == 0:  # only inanimate
        class_label = 'inanimates'

    # mixed non-people and inanimates
    else:
        class_label = 'animates_inanimates'

    class_labels.append(class_label)

# n_images = len(sample)
# all_conditions = range(n_images)

# load RDMs
rdms = []
for mask_name in ROIS:
    rdm_file = os.path.join(
        outpath, f'{sub}_{mask_name}_fullrdm_correlation_session-{n_sessions}.npy'
    )
    print(f'loading full rdm for {mask_name} : {sub}')
    rdm = np.load(rdm_file, allow_pickle=True)
    rdms.append(rdm.astype(np.float32))

# get the sample images
sample_im = nsda.read_images(list(sample-1))

# loop over rois
for roi_i, roi in enumerate(ROIS):

    tsne_fig_file = os.path.join(
        tsne_figures, f'{sub}_{roi}_tsne_session-{n_sessions}.png'
    )
    tsne_fig_file_dots = os.path.join(
        tsne_figures, f'{sub}_{roi}_tsne_dots_session-{n_sessions}.png'
    )
    mds_fig_file_dots = os.path.join(
        tsne_figures, f'{sub}_{roi}_mds_dots_session-{n_sessions}.png'
    )

    # === MDS 
    print(f"Computing MDS embedding for {sub}\n\t {roi}")
    start_time = time.time()
    Y_mds = mds(rdms[roi_i])
    elapsed_time = time.time() - start_time
    print(
        'elapsedtime: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    )

    scprep.plot.scatter2d(
        Y_mds,
        c=class_labels,
        figsize=(8, 8),
        cmap=cate_colors,
        ticks=False,
        legend_loc='lower left',
        legend_ncol=2,
        label_prefix="MDS"
    )

    plt.savefig(mds_fig_file_dots, dpi=400)
    plt.close('all')
    print("saved MDS image!")

    # === tSNE
    print(f"Computing t-SNE embedding for {sub}\n\t {roi}")
    start_time = time.time()
    tsne_operator = manifold.TSNE(
        metric='precomputed',
        perplexity=100,
        n_components=2,
        init=Y_mds,
        n_jobs=n_jobs
    )

    Y_tsne = tsne_operator.fit_transform(squareform(rdms[roi_i]))
    elapsed_time = time.time() - start_time
    print(
        'elapsedtime: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    )

    print("plotting tSNE with dots...")
    scprep.plot.scatter2d(
        Y_tsne,
        c=class_labels,
        figsize=(8, 8),
        cmap=cate_colors,
        ticks=False,
        legend_loc='lower left',
        legend_ncol=2,
        label_prefix="t-SNE")

    plt.savefig(tsne_fig_file_dots, dpi=400)
    plt.close('all')
    print("saved tSNE image with dots!")

    # also plot the figure with all pictures
    print("plotting tSNE with images...")
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    # extent : scalars (left, right, bottom, top)
    scaler = 0.27
    # lets say you have 40 images, first 20 are animate
    for i, pat in enumerate(Y_tsne):
        x, y = pat
        # plot image
        im = sample_im[i, :, :, :]

        ax.imshow(
            im,
            aspect='auto',
            extent=(
                x-(0.92*scaler),
                x+(0.92*scaler),
                y-scaler,
                y+scaler
            ),
            zorder=1
        )

    ax.set_xlim([Y_tsne[:,0].min()-0.5, Y_tsne[:,0].max()+0.5])
    ax.set_ylim([Y_tsne[:,1].min()-0.5, Y_tsne[:,1].max()+0.5])
    ax.set_axis_off()

    plt.savefig(tsne_fig_file, dpi=400)
    plt.close('all')
    print("saved tSNE image with images!")

    # # now cycle through the categories

    # category_figures = os.path.join(
    #         outpath, 'category_figures', sub, roi
    # )

    # # cycle through categories
    # for cat_i, category in enumerate(category_classes):
    #     category_fig_file_dots = os.path.join(
    #         category_figures, f'{sub}_{cat_i:03d}_{all_labels[cat_i]}_dots.png'
    #     )
    #     scprep.plot.scatter2d(
    #         Y_tsne,
    #         c=category,
    #         figsize=(8, 8),
    #         cmap=[[.85, .85, .85], list(category_colors[cat_i][:3])],
    #         ticks=False,
    #         label_prefix="t-SNE")

    #     plt.savefig(category_fig_file_dots, dpi=400, quality=95)
    #     plt.close('all')
