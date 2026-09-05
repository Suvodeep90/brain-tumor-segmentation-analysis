#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tqdm import tqdm
import os
from os import listdir
import time
from random import randint
from os.path import isfile, join
 
import gc 
import numpy as np
from scipy import stats
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py

from skimage import feature

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage

# from IPython.display import Image as show_gif
# from IPython.display import clear_output
# from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

# !pip install opencv-python==4.6.0.66
# !pip install -U albumentations --no-binary qudida,albumentations
import albumentations as A
# from albumentations.pytorch import ToTensor, ToTensorV2


from albumentations import Compose, HorizontalFlip
# from albumentations.pytorch import ToTensor, ToTensorV2 


from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
import multiprocessing

import warnings
warnings.simplefilter("ignore")


# # Definations and descriptions of GLCM Texture Features
# 
# ## 1. Contrast
# - **Description**: An image divided in half, one side very bright and the other very dark.
# - **Formula**: \(\sum_{i,j=0}^{levels-1} (i - j)^2 \times M(i, j)\)
# - **Definition**: Measures the local variations in the GLCM. High contrast values indicate large variations in intensity levels.
# - **Significance**: Represents high contrast, indicating large variations in intensity levels.
# 
# ## 2. Dissimilarity
# - **Description**: An image with gradual texture changes.
# - **Formula**: \(\sum_{i,j=0}^{levels-1} |i - j| \times M(i, j)\)
# - **Definition**: Similar to contrast but gives less weight to intensity differences that are further apart.
# - **Significance**: Illustrates the concept of dissimilarity, where changes in texture are less abrupt.
# 
# ## 3. Homogeneity
# - **Description**: A smooth or blurred image.
# - **Formula**: \(\sum_{i,j=0}^{levels-1} \frac{M(i, j)}{1 + (i - j)^2}\)
# - **Definition**: Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal.
# - **Significance**: Indicates a high degree of homogeneity, where intensity transitions are subtle.
# 
# ## 4. Energy (ASM)
# - **Description**: An image with a constant or very repetitive texture.
# - **Formula**: \(\sum_{i,j=0}^{levels-1} M(i, j)^2\)
# - **Definition**: Provides the sum of squared elements in the GLCM, a measure of textural uniformity.
# - **Significance**: Demonstrates high energy or Angular Second Moment, indicative of textural uniformity.
# 
# ## 5. Entropy
# - **Description**: A highly detailed and complex image, like a forest scene.
# - **Formula**: \(-\sum_{i,j=0}^{levels-1} M(i, j) \log(M(i, j))\)
# - **Definition**: Measures the randomness in the image texture. Higher values indicate more complexity or randomness.
# - **Significance**: Exemplifies high entropy, reflecting complexity or randomness in the image texture.
# 

# # Function to Calculate Volume of a Tumor based on mask file

# In[6]:


def preprocess_mask_labels(mask):
    # whole tumour
    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1
    mask_WT[mask_WT == 2] = 1
    mask_WT[mask_WT == 3] = 1
    # include all tumours 

    # NCR / NET - LABEL 1
    mask_TC = mask.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 3] = 1
    # exclude 2 / 4 labelled tumour 

    # ET - LABEL 4 
    mask_ET = mask.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 3] = 1
    # exclude 2 / 1 labelled tumour 

    mask = np.stack([mask_WT, mask_TC, mask_ET, mask_ET])
    
    return mask 

def read_MRI(dataset, patient_id):
    if dataset == 'Brats2020':
        baseloc = '../input/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
        pefix = 'BraTS20_Training_' + patient_id + '/' + 'BraTS20_Training_' + patient_id
        suffixs = ['_flair.nii','_t2.nii', '_t1.nii', '_t1ce.nii', '_seg.nii']
    elif dataset == 'Brats2023':
        baseloc = '../input/Brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/'
        pefix = 'BraTS-GLI-' + patient_id + '/' + 'BraTS-GLI-' + patient_id
        suffixs = ['-t2f.nii.gz','-t2w.nii.gz', '-t1n.nii.gz', '-t1c.nii.gz', '-seg.nii.gz']

    sample_filename1 = baseloc + pefix + suffixs[0]
    sample_img1_f = nib.load(sample_filename1)
    sample_img1 = np.asarray(sample_img1_f.dataobj)
    # sample_img1 = np.rot90(sample_img1)

    sample_filename2 = baseloc + pefix + suffixs[1]
    sample_img2_f = nib.load(sample_filename2)
    sample_img2 = np.asarray(sample_img2_f.dataobj)
    # sample_img2  = np.rot90(sample_img2)

    sample_filename3 = baseloc + pefix + suffixs[2]
    sample_img3_f = nib.load(sample_filename3)
    sample_img3 = np.asarray(sample_img3_f.dataobj)
    # sample_img3  = np.rot90(sample_img3)

    sample_filename4 = baseloc + pefix + suffixs[3]
    sample_img4_f = nib.load(sample_filename4)
    sample_img4 = np.asarray(sample_img4_f.dataobj)
    # sample_img4  = np.rot90(sample_img4)

    sample_filename_mask = baseloc + pefix + suffixs[4]
    sample_mask_f = nib.load(sample_filename_mask)
    sample_mask = np.asarray(sample_mask_f.dataobj)
    
    return sample_img1, sample_img2, sample_img3, sample_img4, sample_mask 

def calculate_bounding_box(data):
    """
    Calculate the bounding box for the region of interest in a 3D MRI mask file.

    :param data: The MRI image mask.
    :return: roi_start, roi_end coordinates.
    """

    # Find the indices where the tumor is present
    indices = np.array(np.where(data == 1))

    # Calculate the bounding box
    roi_start = np.min(indices, axis=1)
    roi_end = np.max(indices, axis=1) + 1  # Add 1 to include the end index

    return tuple(roi_start), tuple(roi_end)


def crop_mri_mask(data, roi_start, roi_end):
    """
    Crop a region of interest from a 3D MRI mask file.

    :param data: The MRI image mask.
    :param roi_start: The start coordinates (x, y, z) of the ROI.
    :param roi_end: The end coordinates (x, y, z) of the ROI.
    :return: Cropped MRI data.
    """
    # Crop the data
    cropped_data = data[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]

    return cropped_data

def saperate_tumor_and_background(sample_img1, sample_img2, sample_img3, sample_img4, mask):
    masks = preprocess_mask_labels(mask)
    mask_WT, mask_TC, mask_ET = masks[0], masks[1], masks[2]
    
    tumor_only_sample_img1 = np.where(mask_WT == 1, sample_img1, 0)
    tumor_blacked_sample_img1 = np.where(mask_WT == 1, 0, sample_img1)

    tumor_only_sample_img2 = np.where(mask_WT == 1, sample_img2, 0)
    tumor_blacked_sample_img2 = np.where(mask_WT == 1, 0, sample_img2)

    tumor_only_sample_img3 = np.where(mask_WT == 1, sample_img3, 0)
    tumor_blacked_sample_img3 = np.where(mask_WT == 1, 0, sample_img3)

    tumor_only_sample_img4 = np.where(mask_WT == 1, sample_img4, 0)
    tumor_blacked_sample_img4 = np.where(mask_WT == 1, 0, sample_img4)
    return {'tumor_only': [tumor_only_sample_img1, 
                           tumor_only_sample_img2, 
                           tumor_only_sample_img3, 
                           tumor_only_sample_img4], 
            'tumor_blacked': [tumor_blacked_sample_img1, 
                              tumor_blacked_sample_img2, 
                              tumor_blacked_sample_img3, 
                              tumor_blacked_sample_img4]}

def calculate_overall_texture(dataset, patient_id):
    sample_img1, sample_img2, sample_img3, sample_img4, mask  = read_MRI(dataset, patient_id)
    
    stacked_image = {'t1':sample_img1, 
                    't1ce':sample_img2, 
                    't2':sample_img3, 
                    'flair':sample_img4}
    
    image_textures = {}
    for modality in stacked_image.keys():
        image = stacked_image[modality]
    
        # Convert to uint8 for texture analysis
        sample_img_uint8 = (image / image.max() * 255).astype(np.uint8)
        contrast = {}
        dissimilarity = {}
        homogeneity = {}
        energy = {}
        correlation = {}
        ASM = {}
        for _slice in range(image.shape[2]):
            # Compute GLCM and texture properties for a slice
            slice_sample_img = sample_img_uint8[:, :, _slice]
            glcm = feature.graycomatrix(slice_sample_img, 
                                        distances=np.linspace(1, 10, 10),
                                        angles=[0, 3.14/4, 3.14/2,  (3*3.14)/4], 
                                        symmetric=True, 
                                        normed=True)


            # Compute texture properties
            contrast[_slice] = feature.graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity[_slice] = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity[_slice] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            energy[_slice] = feature.graycoprops(glcm, 'energy')[0, 0]
            correlation[_slice] = feature.graycoprops(glcm, 'correlation')[0, 0]
            ASM[_slice] = feature.graycoprops(glcm, 'ASM')[0, 0]
        
        image_textures[modality] = {'contrast': contrast, 
                                    'dissimilarity': dissimilarity, 
                                    'homogeneity': homogeneity, 
                                    'energy': energy, 
                                    'correlation':correlation, 
                                    'ASM':ASM}
    
    
    return image_textures


# # Calculate the Textures of the Image 

# In[7]:


def process_patient(patient_id):
    dataset = 'Brats2023'
    try:
        patient_id_processed = patient_id.split('GLI-')[1]
        return patient_id, calculate_overall_texture(dataset, patient_id_processed)
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        return patient_id, None


if __name__ == '__main__':
    data_path = '../input/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/'
    patient_ids = [f for f in listdir(data_path) if not isfile(join(data_path, f))]
    num_processes = multiprocessing.cpu_count()

    image_textures = {}

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_patient, patient_ids), total=len(patient_ids)))
        
    image_textures = {patient_id: texture for patient_id, texture in results if texture is not None}

    with open('../Results/Analysis_Results/Texture_Analysis.pkl', 'wb') as handle:
        pkl.dump(image_textures, handle, protocol=pkl.HIGHEST_PROTOCOL)


