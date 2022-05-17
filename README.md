# nsddatapaper_rsa (extension of original analyses)

## Extension of rsa analyses for the data paper

This repo (forked from the original rsa analyses) contains modified scripts and additional analyses that stems from the paper, focusing on characterizing the represenations in the parietal lobe and comparing the selectivity between the dorsal stream and the ventral stream. 

Overview of analyses steps: 

* Single-trial GLM (betas provided in NSD dataset);
* RSA (representational dissimilarity matrices, RDMs, from all pairs condition activity patterns);
* Multidimensional scaling (MDS) & tSNE

*This repo contains an ipynp for scratch analyses, presentations slides, and result figures, which may take a lot of space/time to download. We recommend partial clone or sparse chekout for the scripts olnly.*

## Installation notes from original repo
to install simply clone this repo, cd to it's directory and 

```bash
python setupy.py develop
```

this will install the package and all other [required packages](requirements.txt).


### additionally, you may need to install Tomas Knapen's nsd_access

```bash
pip install git+https://github.com/tknapen/nsd_access.git
```

nsd_access is a convenient tool for quickly and easily accessing data from the 
Natural Scenes Dataset. For more details and tutorials see the repo here: https://github.com/tknapen/nsd_access


It might be worth familiarising yourself with the Natural Scenes Dataset before you begin too.
You can do so with the [NSD Data Manual](https://cvnlab.slite.com/app/channels/CT9Fwl4_hc)

## Data Setup
Follow the instructions [in the NSD Manual](https://cvnlab.slite.com/p/channel/CPyFRAyDYpxdkPK6YbB5R1/notes/dC~rBTjqjb) to download the data needed from AWS (create free account; install AWS CLI). The dataset should be contained in a directory named "NSD" parallel to this repo. 

## Script Pipeline

### First, compute the category labels with

```bash
python category_labels.py
```
This saves the category labels for specfied subjects for specified number of sessions in one file, for faster access in later processing. 

### Second, prepare the masked betas in rois along the ventral stream with

```bash
python rois_rdms.py
```
Load betas for specified sessions; filter for trials with 3 presentations; fitler for good voxels; compute & save RDMs.

### Finally, plot the TSNE with

```bash
python plot_tsne.py
```
Compute tSNE, plot & save tSNE plots with dots/images. 

### NEW: group COCO categories with ROI activities with

```bash
python category_grouping.py
```
Currently running PCA on customized voxel_category "betas" matrix, plot & save category geometry in space formed with the first 3 PCs.