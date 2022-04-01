import argparse
import os.path as op
import numpy as np
from nsd_access import NSDAccess
from utils.nsd_get_data import get_conditions

parser = argparse.ArgumentParser()
parser.add_argument("n_subs", help="number of subjects to compute", type=int, default=1)
parser.add_argument("n_sessions", help="n_sessions to load", type=int, default=10)
parser.add_argument("n_jobs", help="n_jobs to run", type=int, default=1)
args = parser.parse_args()

n_subjects = args.n_subs
n_sessions = args.n_sessions
n_jobs = args.n_jobs

# n_sessions = 10
# n_subjects = 1
# n_jobs = 8

# subjects
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

# setup some directories
base_dir = "/work2/07365/sguo19/stampede2/"
nsd_dir = op.join(base_dir, 'NSD')
proj_dir = op.join(base_dir, 'nsddatapaper_rsa')
betas_dir = op.join(proj_dir, 'rsa')

# prepare outputs
outpath = op.join(proj_dir, 'rsa')

# conditions
save_stim = op.join(outpath, f'all_subs_stims_full_session-{n_sessions}.npy')
if not op.exists(save_stim):
    # get unique list of conditions
    conditions_all = []
    for sub in subs:
        # extract conditions data
        conditions = get_conditions(nsd_dir, sub, n_sessions)
        # we also need to reshape conditions to be ntrials x 1
        conditions = np.asarray(conditions).ravel()
        # append across subjects
        conditions_all.append(conditions)

    # this is the 73K format condition name for all 213 000 trials
    # across subjects
    conditions_all = np.concatenate(conditions_all, axis=0)

    # this is as above, but without repeats.
    # useful later when we average across trials for each cond
    conditions = np.unique(conditions_all)

    # save the conditions
    np.save(save_stim, conditions)
else:
    print("loading from existing file:", save_stim)
    # get the condition list
    conditions = np.load(op.join(
        # betas_dir, 'all_subs_stims_full.npy'
        save_stim
    ), allow_pickle=True)


# now that we have all the conditions, let's read in the
# category labels.
nsda = NSDAccess(nsd_dir)

# in nsda we have a module that extracts the categories
# from the annotation files provided with MS coco.
categories = nsda.read_image_coco_category(
        conditions-1, n_jobs=n_jobs
    )

# and save them
np.save(op.join(betas_dir, f'all_stims_category_labels_session-{n_sessions}.npy'), categories)
