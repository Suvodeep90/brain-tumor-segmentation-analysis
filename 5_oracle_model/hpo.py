#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from scipy.stats import wilcoxon, mannwhitneyu
import pickle as pkl
import random

import nibabel as nib

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, SequentialFeatureSelector

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from hpsklearn import HyperoptEstimator, svc

from pytorch_tabnet.tab_model import TabNetRegressor

import xgboost as xgb


# In[2]:


def load_unet_result(path, _print):
    unet_df = pd.read_csv(path, index_col = 'Unnamed: 0')
    index_values = []
    for _index in unet_df.index:
        index_values.append(_index.split('-seg')[0])

    unet_df.index = index_values  
    unet_df.drop(['WT jaccard', 'TC jaccard', 'ET jaccard'], axis = 1, inplace = True)
    summary_unet_df = pd.DataFrame(zip(unet_df.mean().values.tolist(), 
                                       unet_df.std().values.tolist()), 
                                   columns = ['mean', 'std'], index = unet_df.columns)
    if _print:
        print("****UNet******")
        print(summary_unet_df)
    return summary_unet_df, unet_df

def load_nnunet_result(path, _print):
    with open(path, 'r') as file:
        data = json.load(file)

    WT = []
    TC = []
    ET = []
    file_name = []
    for case in data['metric_per_case']:
        WT.append(case['metrics']['(2, 1, 3)']['Dice'])
        TC.append(case['metrics']['(2, 3)']['Dice'])
        ET.append(case['metrics']['(3,)']['Dice'])
        file_name.append(case['reference_file'].split('/')[-1].split('.')[0])

    nnunet_df = pd.DataFrame(zip(WT, TC, ET), columns = ['WT dice', 'TC dice', 'ET dice'], 
                             index = file_name)
    summary_nnunet_df = pd.DataFrame(zip(nnunet_df.mean().values.tolist(), 
                                       nnunet_df.std().values.tolist()), 
                                   columns = ['mean', 'std'], index = nnunet_df.columns)
    if _print:
        print("****nnUNet******")
        print(summary_nnunet_df)
    return summary_nnunet_df, nnunet_df

def load_TransBTS_result(path, _print):
    with open(path, 'r') as file:
        data = json.load(file)

    WT = []
    TC = []
    ET = []
    file_name = []
    for case_id in data.keys():
        case = data[case_id]
        WT.append(case['WT'][0])
        TC.append(case['TC'][0])
        ET.append(case['ET'][0])
        file_name.append(case_id)

    TransBTS_df = pd.DataFrame(zip(WT, TC, ET), columns = ['WT dice', 'TC dice', 'ET dice'], 
                             index = file_name)
    summary_TransBTS_df = pd.DataFrame(zip(TransBTS_df.mean().values.tolist(), 
                                       TransBTS_df.std().values.tolist()), 
                                   columns = ['mean', 'std'], index = TransBTS_df.columns)
    if _print:
        print("****TransBTS******")
        print(summary_TransBTS_df)
    return summary_TransBTS_df, TransBTS_df

def read_radiomics_results(analysis_type, location):
    file_name = '../Results/Analysis_Results/Radiomics/' + location + '/' + analysis_type + '.pkl'
    with open(file_name, 'rb') as f:
        results = pkl.load(f)
    return results

def read_results(_print=True):
    path = '../Results/Result/Vanilla_Unet/Unet_test_dice.csv'
    summary_unet_df, unet_df = load_unet_result(path, _print)

    path = '../Results/Result/nnUnet/nnUNetTrainer/summary.json'
    summary_da_nnunet_df, nnunet_da_df = load_nnunet_result(path, _print)

    path = '../Results/Result/nnUnet/nnUNetTrainerNoDA/summary.json'
    summary_noda_nnunet_df, nnunet_noda_df = load_nnunet_result(path, _print)

    path = '../Results/Result/TransBTS/submission/TransBTS2023-11-03/TransBTS_summary.json'
    summary_TransBTS_df, TransBTS_df = load_TransBTS_result(path, _print)
    return unet_df, nnunet_noda_df, nnunet_da_df, TransBTS_df


def get_dataset(performance_df, analysis_types, location):
    results_df = performance_df
    for analysis_type in analysis_types:
        try:
            radiomics_results_df = read_radiomics_results(analysis_type, location)
        except Exception as e:
            print(e)
            continue
        properties = {}
        property_df = pd.DataFrame()
        for i in range(len(radiomics_results_df.keys())):
            key = list(radiomics_results_df.keys())[i]
            properties[i] = key

        for i in range(len(properties)):
            selected_property = properties[i]

            property_result_df = pd.DataFrame.from_dict(radiomics_results_df[selected_property], 
                                                        orient = 'index').astype(float)
            new_col = []
            for col in property_result_df.columns:
                new_col.append(selected_property + '_' + col + '_' + analysis_type)
            property_result_df.columns = new_col
            results_df = pd.merge(property_result_df, 
                           results_df, 
                           left_index=True, 
                           right_index=True)
#             print(property_result_df.shape)

    remove_index = results_df.sort_values(['WT dice'])[0:20].index
    low_examples = results_df.loc[remove_index]
    results_df.drop(['WT dice', 'TC dice', 'ET dice'], axis=1, inplace=True)
    # results_df.drop(remove_index, axis = 0, inplace=True)
    
    return results_df, low_examples

def get_overlaps(unet_df, TransBTS_df, nnunet_noda_df, WT_dice_threshold, TC_dice_threshold, ET_dice_threshold):
    unet_df_sub = unet_df[(unet_df['WT dice'] < WT_dice_threshold) 
                          & (unet_df['TC dice'] < TC_dice_threshold) 
                          & (unet_df['ET dice'] < ET_dice_threshold)]
    
    TransBTS_df_sub = TransBTS_df[(TransBTS_df['WT dice'] < WT_dice_threshold) 
                          & (TransBTS_df['TC dice'] < TC_dice_threshold) 
                          & (TransBTS_df['ET dice'] < ET_dice_threshold)]
    
    nnunet_noda_df_sub = nnunet_noda_df[(nnunet_noda_df['WT dice'] < WT_dice_threshold) 
                          & (nnunet_noda_df['TC dice'] < TC_dice_threshold) 
                          & (nnunet_noda_df['ET dice'] < ET_dice_threshold)]

    unet_df_sub_subjects = unet_df_sub.index.values.tolist()
    TransBTS_df_sub_subjects = TransBTS_df_sub.index.values.tolist()
    nnunet_noda_df_sub_subjects = nnunet_noda_df_sub.index.values.tolist()

    all_overlaps = list(set(unet_df_sub_subjects) & set(TransBTS_df_sub_subjects) & set(nnunet_noda_df_sub_subjects))
#     print('all overlap', len(all_overlaps), unet_df_sub.shape, nnunet_noda_df_sub.shape)

    unet_nnunet_overlaps = list(set(unet_df_sub_subjects) & set(nnunet_noda_df_sub_subjects))
#     print('unet-nnunet overlap', len(unet_nnunet_overlaps))

    unet_TransBTS_overlaps = list(set(unet_df_sub_subjects) & set(TransBTS_df_sub_subjects))
#     print('unet-TransBTS overlap', len(unet_TransBTS_overlaps))

    nnunet_TransBTS_overlaps = list(set(TransBTS_df_sub_subjects) & set(nnunet_noda_df_sub_subjects))
#     print('nnunet-TransBTS overlap', len(nnunet_TransBTS_overlaps))
    
    return all_overlaps, unet_nnunet_overlaps


# In[3]:


analysis_types = ['firstorder', 'shape' , 'size',
                  'glcm_1', 'glcm_5', 'glcm_10', 
                  'gldm_1', 'gldm_5', 'gldm_10', 
                  'glrlm', 'glszm', 'intensity', 
                  'ngtdm_1', 'ngtdm_5','ngtdm_10']

WT_dice_threshold= 0.91
TC_dice_threshold= 0.86
ET_dice_threshold= 0.85

unet_df, nnunet_noda_df, nnunet_da_df, TransBTS_df = read_results(False)
performance_df = unet_df

all_overlaps, unet_nnunet_overlaps = get_overlaps(unet_df, 
                                                TransBTS_df, 
                                                nnunet_noda_df, 
                                                WT_dice_threshold, 
                                                TC_dice_threshold, 
                                                ET_dice_threshold)


# In[4]:


location = 'Tumor_WT'


# # Shape

# In[5]:


shape_features = ["original_shape_Elongation_flair_shape",
            "original_shape_Flatness_flair_shape",
            "original_shape_LeastAxisLength_flair_shape",
            "original_shape_MajorAxisLength_flair_shape",
            "original_shape_MinorAxisLength_flair_shape",
            "original_shape_Sphericity_flair_shape"]

analysis_types = ['shape']


shape_df, _ = get_dataset(performance_df, analysis_types, location)
shape_df = shape_df[shape_features]
shape_df.shape


# # Size

# In[6]:


size_features = ["diagnostics_Mask-original_VolumeNum_flair_size",
            "original_shape_MeshVolume_flair_size",
            "original_shape_SurfaceArea_flair_size",
            "original_shape_SurfaceVolumeRatio_flair_size"]

analysis_types = ['size']

size_df, _ = get_dataset(performance_df, analysis_types, location)
size_df = size_df[size_features]
size_df.shape


# # Intensity

# In[7]:


intensity_features = ["diagnostics_Image-original_Mean_flair_intensity",
            "diagnostics_Image-original_Mean_t2_intensity",
            "diagnostics_Image-original_Mean_t1_intensity",
            "diagnostics_Image-original_Mean_t1ce_intensity",
            "diagnostics_Image-original_Maximum_flair_intensity",
            "diagnostics_Image-original_Maximum_t2_intensity",
            "diagnostics_Image-original_Maximum_t1_intensity",
            "diagnostics_Image-original_Maximum_t1ce_intensity"]

analysis_types = ['intensity']

intensity_df, _ = get_dataset(performance_df, analysis_types, location)
intensity_df = intensity_df[intensity_features]
intensity_df.shape


# # First Order Statistics

# In[8]:


firstorder_features = ["original_firstorder_Energy_flair_firstorder",
            "original_firstorder_Energy_t2_firstorder",
            "original_firstorder_Energy_t1_firstorder",
            "original_firstorder_Energy_t1ce_firstorder",
            "original_firstorder_Entropy_flair_firstorder",
            "original_firstorder_Entropy_t2_firstorder",
            "original_firstorder_Entropy_t1_firstorder",
            "original_firstorder_Entropy_t1ce_firstorder",
            "original_firstorder_Kurtosis_flair_firstorder",
            "original_firstorder_Kurtosis_t2_firstorder",
            "original_firstorder_Kurtosis_t1_firstorder",
            "original_firstorder_Kurtosis_t1ce_firstorder",
            "original_firstorder_Skewness_flair_firstorder",
            "original_firstorder_Skewness_t2_firstorder",
            "original_firstorder_Skewness_t1_firstorder",
            "original_firstorder_Skewness_t1ce_firstorder",
            "original_firstorder_Uniformity_flair_firstorder",
            "original_firstorder_Uniformity_t2_firstorder",
            "original_firstorder_Uniformity_t1_firstorder",
            "original_firstorder_Uniformity_t1ce_firstorder"]

analysis_types = ['firstorder']

firstorder_df, _ = get_dataset(performance_df, analysis_types, location)
firstorder_df = firstorder_df[firstorder_features]
firstorder_df.shape


# # Neighbouring Gray Tone Difference Matrix (NGTDM) features

# In[9]:


ngtdm_features = ["original_ngtdm_Busyness_flair_ngtdm_10",
        "original_ngtdm_Busyness_t2_ngtdm_10",
        "original_ngtdm_Busyness_t1_ngtdm_10",
        "original_ngtdm_Busyness_t1ce_ngtdm_10",
        "original_ngtdm_Coarseness_flair_ngtdm_10",
        "original_ngtdm_Coarseness_t2_ngtdm_10",
        "original_ngtdm_Coarseness_t1_ngtdm_10",
        "original_ngtdm_Coarseness_t1ce_ngtdm_10",
        "original_ngtdm_Complexity_flair_ngtdm_10",
        "original_ngtdm_Complexity_t2_ngtdm_10",
        "original_ngtdm_Complexity_t1_ngtdm_10",
        "original_ngtdm_Complexity_t1ce_ngtdm_10",
        "original_ngtdm_Contrast_flair_ngtdm_10",
        "original_ngtdm_Contrast_t2_ngtdm_10",
        "original_ngtdm_Contrast_t1_ngtdm_10",
        "original_ngtdm_Contrast_t1ce_ngtdm_10",
        "original_ngtdm_Strength_flair_ngtdm_10",
        "original_ngtdm_Strength_t2_ngtdm_10",
        "original_ngtdm_Strength_t1_ngtdm_10",
        "original_ngtdm_Strength_t1ce_ngtdm_10", ]


analysis_types = ['ngtdm_10']

ngtdm_df, _ = get_dataset(performance_df, analysis_types, location)
ngtdm_df = ngtdm_df[ngtdm_features]
ngtdm_df.shape


# # Gray Level Co-occurrence Matrix (GLCM) Features

# In[10]:


glcm_features = ["original_glcm_Autocorrelation_flair_glcm_10",
        "original_glcm_Autocorrelation_t2_glcm_10",
        "original_glcm_Autocorrelation_t1_glcm_10",
        "original_glcm_Autocorrelation_t1ce_glcm_10",
        "original_glcm_ClusterProminence_flair_glcm_10",
        "original_glcm_ClusterProminence_t2_glcm_10",
        "original_glcm_ClusterProminence_t1_glcm_10",
        "original_glcm_ClusterProminence_t1ce_glcm_10",
        "original_glcm_ClusterShade_flair_glcm_10",
        "original_glcm_ClusterShade_t2_glcm_10",
        "original_glcm_ClusterShade_t1_glcm_10",
        "original_glcm_ClusterShade_t1ce_glcm_10",
        "original_glcm_ClusterTendency_flair_glcm_10",
        "original_glcm_ClusterTendency_t2_glcm_10",
        "original_glcm_ClusterTendency_t1_glcm_10",
        "original_glcm_ClusterTendency_t1ce_glcm_10",
        "original_glcm_Contrast_flair_glcm_10",
        "original_glcm_Contrast_t2_glcm_10",
        "original_glcm_Contrast_t1_glcm_10",
        "original_glcm_Contrast_t1ce_glcm_10",
        "original_glcm_Correlation_flair_glcm_10",
        "original_glcm_Correlation_t2_glcm_10",
        "original_glcm_Correlation_t1_glcm_10",
        "original_glcm_Correlation_t1ce_glcm_10",
        "original_glcm_JointAverage_flair_glcm_10",
        "original_glcm_JointAverage_t2_glcm_10",
        "original_glcm_JointAverage_t1_glcm_10",
        "original_glcm_JointAverage_t1ce_glcm_10",
        "original_glcm_JointEnergy_flair_glcm_10",
        "original_glcm_JointEnergy_t2_glcm_10",
        "original_glcm_JointEnergy_t1_glcm_10",
        "original_glcm_JointEnergy_t1ce_glcm_10",
        "original_glcm_JointEntropy_flair_glcm_10",
        "original_glcm_JointEntropy_t2_glcm_10",
        "original_glcm_JointEntropy_t1_glcm_10",
        "original_glcm_JointEntropy_t1ce_glcm_10",
        "original_glcm_MCC_flair_glcm_10",
        "original_glcm_MCC_t2_glcm_10",
        "original_glcm_MCC_t1_glcm_10",
        "original_glcm_MCC_t1ce_glcm_10"]


analysis_types = ['glcm_10']

glcm_df, _ = get_dataset(performance_df, analysis_types, location)
glcm_df = glcm_df[glcm_features]
glcm_df.shape


# # Gray Level Dependence Matrix (GLDM) Features

# In[11]:


gldm_features = ["original_gldm_DependenceNonUniformity_flair_gldm_10",
        "original_gldm_DependenceNonUniformity_t2_gldm_10",
        "original_gldm_DependenceNonUniformity_t1_gldm_10",
        "original_gldm_DependenceNonUniformity_t1ce_gldm_10",
        "original_gldm_GrayLevelNonUniformity_flair_gldm_10",
        "original_gldm_GrayLevelNonUniformity_t2_gldm_10",
        "original_gldm_GrayLevelNonUniformity_t1_gldm_10",
        "original_gldm_GrayLevelNonUniformity_t1ce_gldm_10",
        "original_gldm_GrayLevelVariance_flair_gldm_10",
        "original_gldm_GrayLevelVariance_t2_gldm_10",
        "original_gldm_GrayLevelVariance_t1_gldm_10",
        "original_gldm_GrayLevelVariance_t1ce_gldm_10",
        "original_gldm_HighGrayLevelEmphasis_flair_gldm_10",
        "original_gldm_HighGrayLevelEmphasis_t2_gldm_10",
        "original_gldm_HighGrayLevelEmphasis_t1_gldm_10",
        "original_gldm_HighGrayLevelEmphasis_t1ce_gldm_10",
        "original_gldm_LargeDependenceEmphasis_flair_gldm_10",
        "original_gldm_LargeDependenceEmphasis_t2_gldm_10",
        "original_gldm_LargeDependenceEmphasis_t1_gldm_10",
        "original_gldm_LargeDependenceEmphasis_t1ce_gldm_10",
        "original_gldm_LowGrayLevelEmphasis_flair_gldm_10",
        "original_gldm_LowGrayLevelEmphasis_t2_gldm_10",
        "original_gldm_LowGrayLevelEmphasis_t1_gldm_10",
        "original_gldm_LowGrayLevelEmphasis_t1ce_gldm_10",
        "original_gldm_SmallDependenceEmphasis_flair_gldm_10",
        "original_gldm_SmallDependenceEmphasis_t2_gldm_10",
        "original_gldm_SmallDependenceEmphasis_t1_gldm_10",
        "original_gldm_SmallDependenceEmphasis_t1ce_gldm_10"]


analysis_types = ['gldm_10']

gldm_df, _ = get_dataset(performance_df, analysis_types, location)
gldm_df = gldm_df[gldm_features]
gldm_df.shape


# # Gray Level Run Length Matrix (GLRLM) Features

# In[12]:


glrlm_features = ["original_glrlm_GrayLevelNonUniformity_flair_glrlm",
        "original_glrlm_GrayLevelNonUniformity_t2_glrlm",
        "original_glrlm_GrayLevelNonUniformity_t1_glrlm",
        "original_glrlm_GrayLevelNonUniformity_t1ce_glrlm",
        "original_glrlm_LongRunEmphasis_flair_glrlm",
        "original_glrlm_LongRunEmphasis_t2_glrlm",
        "original_glrlm_LongRunEmphasis_t1_glrlm",
        "original_glrlm_LongRunEmphasis_t1ce_glrlm",
        "original_glrlm_LongRunHighGrayLevelEmphasis_flair_glrlm",
        "original_glrlm_LongRunHighGrayLevelEmphasis_t2_glrlm",
        "original_glrlm_LongRunHighGrayLevelEmphasis_t1_glrlm",
        "original_glrlm_LongRunHighGrayLevelEmphasis_t1ce_glrlm",
        "original_glrlm_LongRunLowGrayLevelEmphasis_flair_glrlm",
        "original_glrlm_LongRunLowGrayLevelEmphasis_t2_glrlm",
        "original_glrlm_LongRunLowGrayLevelEmphasis_t1_glrlm",
        "original_glrlm_LongRunLowGrayLevelEmphasis_t1ce_glrlm",
        "original_glrlm_LowGrayLevelRunEmphasis_flair_glrlm",
        "original_glrlm_LowGrayLevelRunEmphasis_t2_glrlm",
        "original_glrlm_LowGrayLevelRunEmphasis_t1_glrlm",
        "original_glrlm_LowGrayLevelRunEmphasis_t1ce_glrlm",
        "original_glrlm_RunLengthNonUniformity_flair_glrlm",
        "original_glrlm_RunLengthNonUniformity_t2_glrlm",
        "original_glrlm_RunLengthNonUniformity_t1_glrlm",
        "original_glrlm_RunLengthNonUniformity_t1ce_glrlm",
        "original_glrlm_RunPercentage_flair_glrlm",
        "original_glrlm_RunPercentage_t2_glrlm",
        "original_glrlm_RunPercentage_t1_glrlm",
        "original_glrlm_RunPercentage_t1ce_glrlm",
        "original_glrlm_ShortRunEmphasis_flair_glrlm",
        "original_glrlm_ShortRunEmphasis_t2_glrlm",
        "original_glrlm_ShortRunEmphasis_t1_glrlm",
        "original_glrlm_ShortRunEmphasis_t1ce_glrlm",
        "original_glrlm_ShortRunHighGrayLevelEmphasis_flair_glrlm",
        "original_glrlm_ShortRunHighGrayLevelEmphasis_t2_glrlm",
        "original_glrlm_ShortRunHighGrayLevelEmphasis_t1_glrlm",
        "original_glrlm_ShortRunHighGrayLevelEmphasis_t1ce_glrlm",
        "original_glrlm_ShortRunLowGrayLevelEmphasis_flair_glrlm",
        "original_glrlm_ShortRunLowGrayLevelEmphasis_t2_glrlm",
        "original_glrlm_ShortRunLowGrayLevelEmphasis_t1_glrlm",
        "original_glrlm_ShortRunLowGrayLevelEmphasis_t1ce_glrlm"]


analysis_types = ['glrlm']

glrlm_df, _ = get_dataset(performance_df, analysis_types, location)
glrlm_df = glrlm_df[glrlm_features]
glrlm_df.shape


# # Gray Level Size Zone Matrix (GLSZM) Features

# In[13]:


glszm_features = ["original_glszm_LargeAreaEmphasis_flair_glszm",
        "original_glszm_LargeAreaEmphasis_t2_glszm",
        "original_glszm_LargeAreaEmphasis_t1_glszm",
        "original_glszm_LargeAreaEmphasis_t1ce_glszm",
        "original_glszm_SizeZoneNonUniformity_flair_glszm",
        "original_glszm_SizeZoneNonUniformity_t2_glszm",
        "original_glszm_SizeZoneNonUniformity_t1_glszm",
        "original_glszm_SizeZoneNonUniformity_t1ce_glszm",
        "original_glszm_SmallAreaEmphasis_flair_glszm",
        "original_glszm_SmallAreaEmphasis_t2_glszm",
        "original_glszm_SmallAreaEmphasis_t1_glszm",
        "original_glszm_SmallAreaEmphasis_t1ce_glszm",
        "original_glszm_ZoneEntropy_flair_glszm",
        "original_glszm_ZoneEntropy_t2_glszm",
        "original_glszm_ZoneEntropy_t1_glszm",
        "original_glszm_ZoneEntropy_t1ce_glszm"]


analysis_types = ['glszm']

glszm_df, _ = get_dataset(performance_df, analysis_types, location)
glszm_df = glszm_df[glszm_features]
glszm_df.shape


# # Volume

# In[14]:


volume_df = pd.read_csv('../Results/Analysis_Results/volume/GLI-Tumor_volumns.csv', 
                         index_col='Unnamed: 0')



volume_df = volume_df[['ED', 'ET', 'NCR', 'WT_volume', 'TC_volume', 'ET_volume', 'TC_WT_ratio',
       'ET_WT_ratio', 'ET_TC_ratio']]

volume_df.shape


# # Probability

# In[15]:


Probability_df = pd.read_csv('../Results/Analysis_Results/probability/Probability_Tumor_boundary.csv', 
                             index_col='Unnamed: 0')

Probability_df.shape


# # Curverature

# In[16]:


Curverature_df = pd.read_csv('../Results/Analysis_Results/curverature/curverature.csv', 
                             index_col='Unnamed: 0')

Curverature_df = Curverature_df[['mean_gaussian_curvature', 'std_gaussian_curvature', 
                                'pos', 'neg',
                                'pos_count', 'neg_count']]

Curverature_df.shape


# # Saliency

# In[17]:


Saliency_df = pd.read_csv('../Results/Analysis_Results/Saliency/Saliency.csv', 
                             index_col='Unnamed: 0')

Saliency_df.shape


# # Combine Features

# In[18]:


results_df = pd.merge(shape_df, 
                       size_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       intensity_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       volume_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       Curverature_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       Saliency_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       Probability_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       firstorder_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       ngtdm_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       glcm_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       gldm_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       glrlm_df, 
                       left_index=True, 
                       right_index=True)

results_df = pd.merge(results_df, 
                       glszm_df, 
                       left_index=True, 
                       right_index=True)


results_df = pd.merge(results_df, 
                       performance_df, 
                       left_index=True, 
                       right_index=True)


# In[19]:


results_df = results_df.dropna()
results_df.shape


# # Build Model

# In[ ]:


def train_model(X_train, y_train):
    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, 
                                      learning_rate=0.1, 
                                      max_depth=2, 
                                      random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test):
    # Predict on the testing set
    y_pred = model.predict(X_test)
    return y_pred

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

# def select_features(X_train, y_train):
#     # rfe = RFE(estimator=model, n_features_to_select=20)
#     sfs = SequentialFeatureSelector(estimator = model, 
#                                     n_features_to_select = 15, 
#                                     scoring = 'neg_mean_absolute_error', 
#                                     n_jobs=4)
#     sfs.fit(X_train_scaled, y_train)

#     selected_features = sfs.get_feature_names_out()

#     # Print the selected features
#     print("Selected features:", selected_features)
    
#     return selected_features

def select_features(X_train, y_train):
    # rfe = RFE(estimator=model, n_features_to_select=20)
    mir = mutual_info_regression(X_train_scaled, y_train)
    
    select_df = pd.DataFrame(zip(X_train_scaled.columns, mir), columns = ['features', 'score'])

    selected_features = select_df.sort_values(['score'], ascending=False)[0:50].features.values.tolist()

    # Print the selected features
#     print("Selected features:", selected_features)
    
    return selected_features


# ##  Train Baseline Model

# In[25]:


# Separate features and target variable
results_df.reset_index(drop=True, inplace=True)
X = results_df.drop(['WT dice', 'TC dice', 'ET dice'], axis=1)  # Features
y = results_df['WT dice']  # Target variable (Dice score)


import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functools import partial
import multiprocessing
from deap import tools, algorithms


def customMutate(individual, param_bounds, indpb):
    for i, key in enumerate(param_bounds.keys()):
        if np.random.random() < indpb:  # Probability of mutation
            if isinstance(param_bounds[key], tuple):  # Numeric parameters
                # Mutate within the bounds
                bound = param_bounds[key]
                individual[i] = np.random.uniform(bound[0], bound[1])
            else:  # Categorical parameters
                # Randomly select a new category
                individual[i] = np.random.choice(param_bounds[key])
    return individual,

# Define your `evalModel` function here
def evalModel(individual, X_train, y_train, param_bounds):
    param_names = list(param_bounds.keys())
    params = {param_names[i]: individual[i] for i in range(len(individual))}
    params['n_estimators'] = int(params['n_estimators'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['max_depth'] = int(params['max_depth'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingRegressor(random_state=42, **params)
    mae = -cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
    
    return mae,

def main():
    # Hyperparameter bounds and other setup code...
    # Define the hyperparameter bounds
    param_bounds = {
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'learning_rate': (0.01, 1),
        'n_estimators': (10, 300),
        'subsample': (0.5, 1.0),
        'criterion': ['friedman_mse', 'squared_error'],
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5),
        'min_weight_fraction_leaf': (0.0, 0.5),
        'max_depth': (3, 10),
        'min_impurity_decrease': (0.0, 0.1),
        'max_features': [None, 'sqrt', 'log2'],
        'ccp_alpha': (0.0, 0.1)
    }



    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Attribute generators
    for param, bounds in param_bounds.items():
        if isinstance(bounds, tuple):  # Numeric parameter
            toolbox.register(param, np.random.uniform, bounds[0], bounds[1])
        else:  # Categorical parameter
            toolbox.register(param, np.random.choice, bounds)


    # Individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.loss, toolbox.learning_rate, toolbox.n_estimators, toolbox.subsample,
                      toolbox.criterion, toolbox.min_samples_split, toolbox.min_samples_leaf,
                      toolbox.min_weight_fraction_leaf, toolbox.max_depth, toolbox.min_impurity_decrease,
                      toolbox.max_features, toolbox.ccp_alpha), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Your genetic algorithm code...
    # Make sure to use partial to fix the parameters for evalModel
    # evalModel_partial = partial(evalModel, param_bounds=param_bounds)

    # toolbox.register("evaluate", evalModel_partial)

     # Genetic operators
    toolbox.register("evaluate", evalModel)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    # Register the custom mutation function in the toolbox
    toolbox.register("mutate", customMutate, param_bounds=param_bounds, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Outer K-Fold Cross-validation and other operations...
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores, mae_scores, r2_scores = [], [], []

    for train_index, test_index in kf_outer.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create a partial function of evalModel fixing X_train and y_train
        evalModel_partial = partial(evalModel, X_train=X_train, y_train=y_train, param_bounds=param_bounds)

        # Register the evaluation function with the partial function
        toolbox.register("evaluate", evalModel_partial)
        
        # Create statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        # Genetic Algorithm
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)

        result, log = algorithms.eaSimple(population, toolbox, 
                                          cxpb=0.5, mutpb=0.2, 
                                          ngen=40, stats=stats, 
                                          halloffame=hof, verbose=True)
        
        # # Print the log of each generation
        # for gen in log:
        #     print(f"Generation: {gen['gen']}")
        #     print(f"  Min MAE: {gen['min']}")
        #     print(f"  Max MAE: {gen['max']}")
        #     print(f"  Avg MAE: {gen['avg']}")
        #     print(f"  Std Dev: {gen['std']}\n")

        best_individual = hof[0]
        best_params = {list(param_bounds.keys())[i]: best_individual[i] for i in range(len(best_individual))}
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        best_params['max_depth'] = int(best_params['max_depth'])
        print(best_params)

        # Train the final model on the training data with the best hyperparameters
        model = GradientBoostingRegressor(random_state=42, **best_params)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model on the test set of the current fold
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Average performance across all folds
    print(f"Average MSE: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")
    print(f"Average MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    print(f"Average R2: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

if __name__ == '__main__':
    main()