import os

base_dir = "/work2/07365/sguo19/stampede2/"
nsd_dir = os.path.join(base_dir, 'NSD')
fs_dir = os.path.join(nsd_dir, "nsddata", "freesurfer")

subIDs = [1]
subIDs = [f"subj0{num}" for num in subIDs]
roid = {
    7: "fusiform",
    8: "inferiorparietal", 
    9: "inferiortemporal",
    16: "parahippocampal", 
    25: "precuneus", 
    29: "superiorparietal",
}

for subID in subIDs:
    aparc_dir = os.path.join(fs_dir, subID, "mri")
    aparc_path = aparc_dir + "aparc.DKTatlas+aseg.mgz"
    nifti_dir = aparc_dir
    nifti_path = nifti_dir + "aparc.DKTatlas+aseg.nii.gz"

    ROIs_nifti_dir = os.path.join(nifti_dir, "indiv_roi_niftis")
    if not os.path.exists(ROIs_nifti_dir):
        os.mkdir(ROIs_nifti_dir)
    
    mask_path = os.path.join(fs_dir, subID, "label")

    # convert aparc to nifti:
    if not os.path.exists(nifti_path):
        os.system(f"mri_convert {aparc_path} {nifti_path}")

    # extract rois
    for roi_num, roi_name in roid.items():
        roi_out_path = f"{ROIs_nifti_dir}/{roi_name}.nii.gz"
        if not os.path.exists(roi_out_path):
            os.system(f"fslmaths {nifti_path} -thr {roi_num} -uthr {roi_num+1} {roi_out_path}")

    # # binarize
    # fslmaths pwd/<roi-name>.nii.gz -bin pwd/<roi-name>_bin.nii.gz