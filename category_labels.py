"""
This scipt takes all trialIDs shown to all subjects specified, from all sessions specified, 
and filter for trials that are presented 3 times.

Outputs: 
"stims_full"/conditions: save the unique trialIDs; 
"sample_labels" (in nsd_get_data): 
    generate trial category matrix (onehot of COCO categories) for each subject.
"""
import os.path as op
import numpy as np
from utils.nsd_get_data import get_conditions, get_labels
from config import *

# # other variables needed & specified in config
# n_sessions
# n_jobs

# subjects to include in saved files
n_subjects = 8
subs = ['subj0{}'.format(x+1) for x in range(n_subjects)]

# setup some directories
base_dir = "/work2/07365/sguo19/stampede2/"
nsd_dir = op.join(base_dir, 'NSD')
proj_dir = op.join(base_dir, 'nsddatapaper_rsa')
betas_dir = op.join(proj_dir, 'rsa')

# prepare outputs
outpath = op.join(proj_dir, 'rsa')

# conditions (trialIDs)
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

        # === save category matrix for rdm computation
        # (adding this time-consuming step here so roi_rdm script takes shorter time)
        print(f"making sample category matrix for {sub}...")
        # then we find the valid trials for which we do have 3 repetitions.
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions]
        # find the subject's condition list (sample pool)
        sample = np.unique(conditions[conditions_bool])
        # retrieve the category matrix for the sample & save with funct internal process
        _ = get_labels(sub, betas_dir, nsd_dir, sample-1, n_sessions=n_sessions, n_jobs=n_jobs)

    # this is the 73K format condition name for all 213 000 trials
    # across subjects
    conditions_all = np.concatenate(conditions_all, axis=0)

    # this is as above, but without repeats.
    # useful later when we average across trials for each cond
    conditions = np.unique(conditions_all)

    # save the unique conditions
    np.save(save_stim, conditions)
else:
    print("loading from existing file:", save_stim)
    # get the condition list
    conditions = np.load(save_stim, allow_pickle=True)
