
"""
This script takes the saved RDMs, compute MDS and plot tSNE plots with dots and with actual images.
Uncomment code blocks to plot MDS and single category plots.
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scprep
import time
from matplotlib import cm
from nsd_access import NSDAccess
from utils.nsd_get_data import get_conditions, get_labels
from scipy.spatial.distance import squareform
from sklearn import manifold
from utils.utils import category_dict, mds


def plot_tsne(sub = "subj01",
                n_sessions = 20,
                n_jobs = 1,
                ROIS = ['pVTC', 'aVTC', 'v1', 'v2', 'v3'],
                rdm_dim = "trial",
                color="NSD",
                plot_minor=True,
                plot_tsne_with_figure=[True, True, False, False, False]
                ):
    """
    Input: 
    rdm_dim: possible values: ["trial", "cate"]. Dimension rdm is done on. 
             See rois_rdms.py for more details.
    color: possible values: ["NSD", "minor"]. color theme to use for the categories.  
           If "NSD", follow theme of NSd paper figure 5.
           If "minor", use red for people, green for animals, blue for inanimates, 
           and color combo for cate combo. See ipynb/backup presentation slides for more detials.
    plot_minor: whether to plot binary maps for 6 categories.
    plot_tsne_with_figure: whether to plot tSNE with actual images. This process takes a long time so
                           only do when necessary.
    """
    # ===== set up directories
    base_dir = "/work2/07365/sguo19/stampede2/"
    nsd_dir = os.path.join(base_dir, 'NSD')
    proj_dir = os.path.join(base_dir, 'nsddatapaper_rsa')
    betas_dir = os.path.join(proj_dir, 'rsa')

    outpath = os.path.join(betas_dir, 'roi_analyses', sub)
    figures_outpath = os.path.join(betas_dir, 'roi_analyses')
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # save plots in diff dirs for dorsal & ventral 
    if "pVTC" in ROIS:
        tsne_figures = os.path.join(
                figures_outpath, "ventral_tsne_figures"  #"self_tsne_figures"  #'tsne_figures'
        )
    else: 
        tsne_figures = os.path.join(
                figures_outpath, "dorsal_tsne_figures"  #'tsne_figures'
        )

    if not os.path.exists(tsne_figures):
        os.makedirs(tsne_figures)

    # # for single category plots. Not used for now. 
    # for mask_name in ROIS:
    #     category_figures = os.path.join(
    #             figures_outpath, 'category_figures', sub, mask_name
    #     )

    #     if not os.path.exists(category_figures):
    #         os.makedirs(category_figures)

    # ===== load conditions data
    conditions = get_conditions(nsd_dir, sub, n_sessions)
    conditions = np.asarray(conditions).ravel()  # ntrials x 1
    # then we find the valid trials for which we do have 3 repetitions.
    conditions_bool = [
        True if np.sum(conditions == x) == 3 else False for x in conditions]
    # find the subject's condition list (sample pool)
    sample = np.unique(conditions[conditions_bool])

    # retrieve the category matrix for the sample
    # *** the underlying nsd_access method is modified with Parallel. Original does not take n_jobs ***
    category_matrix = get_labels(sub, betas_dir, nsd_dir, sample-1, n_sessions=n_sessions, n_jobs=n_jobs)

    # ===== get the sample images for tSNE image plotting
    # only load the images if any plotting is needed
    if any(plot_tsne_with_figure): 
        nsda = NSDAccess(nsd_dir)
        print("Loading all images for tSNE with figures...")
        sample_im = nsda.read_images(list(sample-1))

    # # ===== labels for single category binary maps (80 COCO categories)
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

    # ===== colormap
    if color == "minor": 
        # three axis, with red for people, green for animals, blue for inanimates, 
        # and color combo for category combo
        cate_colors = ["red", "gold", "grey", "mediumorchid", "limegreen", "cyan", "blue", ]
    elif color == "NSD": 
        # NSD colors
        cate_colors = ["black", "dimgray", "lightgray", "darkgray", "red", "purple", "blue", ]

    # ===== prepare the class labels
    class_labels = []  # label for each image
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

    # ===== plot each rois
    for roi_i, roi in enumerate(ROIS):
        # save tSNE for replotting with diff category colors
        tsne_file = os.path.join(
            outpath, f'{sub}_{roi}_tsne_session-{n_sessions}.npy'
        )
        tsne_fig_file = os.path.join(
            tsne_figures, f'{sub}_{roi}_tsne_session-{n_sessions}.svg'
        )
        tsne_fig_file_dots = os.path.join(
            tsne_figures, f'{sub}_{roi}_tsne_dots_session-{n_sessions}.png'
        )
        tsne_fig_file_dots_minor_axis = os.path.join(
            tsne_figures, f'{sub}_{roi}_tsne_dots_session-{n_sessions}_axis-minor.png'
        )
        # mds_fig_file_dots = os.path.join(
        #     tsne_figures, f'{sub}_{roi}_mds_dots_session-{n_sessions}.png'
        # )

        if not os.path.exists(tsne_file):
            # load RDM
            if rdm_dim == "trial": 
                rdm_file = os.path.join(
                    outpath, f'{sub}_{roi}_fullrdm_correlation_session-{n_sessions}.npy'
                )
            else: 
                rdm_file = os.path.join(
                    outpath, f'{sub}_{roi}_fullrdm-{rdm_dim}_correlation_session-{n_sessions}.npy'
                )
            print(f'loading full rdm for {roi} : {sub}')
            rdm = np.load(rdm_file, allow_pickle=True)

            # === MDS 
            print(f"Computing MDS embedding for {sub}\n\t {roi}")
            start_time = time.time()
            Y_mds = mds(rdm)
            elapsed_time = time.time() - start_time
            print(
                'elapsedtime: ',
                f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
            )

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

            # save tSNE so they are always the same
            Y_tsne = tsne_operator.fit_transform(squareform(rdm))
            elapsed_time = time.time() - start_time
            print(
                'elapsedtime: ',
                f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
            )

            print(f"saving tsne obj...")
            np.save(tsne_file, Y_tsne)
        else: 
            print("loading from saved tsne file...")
            Y_tsne = np.load(tsne_file)

        # ======= PLOTS ======= 
        # # === plot mds
        # scprep.plot.scatter2d(
        #     Y_mds,
        #     c=class_labels,
        #     figsize=(8, 8),
        #     cmap=cate_colors,
        #     ticks=False,
        #     legend_loc='lower left',
        #     legend_ncol=2,
        #     label_prefix="MDS"
        # )

        # plt.savefig(mds_fig_file_dots, dpi=400)
        # plt.close('all')
        # print("saved MDS image!")

        # === plot tSNE
        print("plotting tSNE with dots...")
        scprep.plot.scatter2d(
            Y_tsne,
            c=class_labels,
            figsize=(5, 5),
            cmap=cate_colors,
            ticks=False,
            # legend_loc='lower left',
            # legend_ncol=2,
            legend=False,
            # label_prefix="t-SNE",
            s=10,
            )
        plt.axis('off')

        plt.savefig(tsne_fig_file_dots, dpi=400)
        plt.close('all')
        print("saved tSNE image with dots!")

        # plot minor axis (binary plots for 6 categories)
        if plot_minor: 
            unique_labels = sorted(set(class_labels))

            fig, axes = plt.subplots(len(unique_labels),1, figsize=(5,4.5*len(unique_labels)))
            for catei, (cate, color) in enumerate(zip(unique_labels, cate_colors)):
                color_ls = ["whitesmoke" for _ in range(len(cate_colors))]
                color_ls[catei] = color
                
                ax = axes[catei]
                scprep.plot.scatter2d(
                    Y_tsne,
                    c=class_labels,
                    # figsize=(5, 5),
                    cmap=color_ls,
                    ticks=False,
                    legend=False,
                    label_prefix="t-SNE",
                    s = 10,
                    title=f"{roi} {cate}",
                    ax=ax  #[1]
                )
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(tsne_fig_file_dots_minor_axis, dpi=400)
            plt.close('all')
            print("saved tSNE image with dots for minor axis!")

        # === also plot tSNE with all pictures
        if plot_tsne_with_figure[roi_i]: 
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

        # === single category binary maps (80 COCO categories)
        # category_figures = os.path.join(
        #         outpath, 'category_figures', sub, roi
        # )

        # # cycle through categories
        # for cat_i, category in enumerate(category_classes):
        #     category_fig_file_dots = os.path.join(
        #         category_figures, f'{sub}_{cat_i:03d}_{all_labels[cat_i]}_dots.png'
        #     )
        #     scprep.plot.scatter2d(n_sessions
        #         figsize=(8, 8),
        #         cmap=[[.85, .85, .85], list(category_colors[cat_i][:3])],
        #         ticks=False,
        #         label_prefix="t-SNE")

        #     plt.savefig(category_fig_file_dots, dpi=400, quality=95)
        #     plt.close('all')

if __name__ == "__main__":
    from config import *

    n_subjects = 8
    subs = ['subj0{}'.format(x+1) for x in range(n_subjects)] 
    ROI_dicts = {
        # group_level: ROIS
        "pathway": ["ventral"],
        "region": ["aVTC", "pVTC", "v1", "v2", "v3"],
        "pathway": ["dorsal"],
        "region": ["IPL", "SPL", "PCC"],
    }
    plot_tsne_with_figures = {
        # corresponding to selections above
        "pathway": [True],
        "region": [True, True, False, False, False],
        "pathway": [True],
        "region": [True, True, True],
    }

    for sub in subs:  
        print(f"\n***** Plotting tSNE for {sub}... *****")
        for (_, ROIS), plot_tsne_with_figure in zip(ROI_dicts.items(), plot_tsne_with_figures):
            print(f"*** running {ROIS} ***")

            plot_tsne(sub=sub,
                        n_sessions=n_sessions,  #from config
                        n_jobs=n_jobs,          #from config
                        ROIS=ROIS,
                        rdm_dim = rdm_dim,      #from config
                        color=color,            #from config
                        plot_minor=plot_minor,  #from config
                        plot_tsne_with_figure=plot_tsne_with_figure
                        )