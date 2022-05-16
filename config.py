"""
This config file contains variables specified for the pipeline
"""
# === overall
sub = "subj01"
n_sessions = 20
n_jobs = 6

# === roi
group_level = "dorsal"
ROIS = ['pVTC', 'aVTC', 'v1', 'v2', 'v3']
rdm_dim = "trial"

# === plot
color = "NSD"
plot_minor = True
plot_tsne_with_figure = [True, True, False, False, False]

# === color category 
n_pcs = 10