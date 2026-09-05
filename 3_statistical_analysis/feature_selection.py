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
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, anneal

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

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

def get_feature_subset(results_df, performance_df):
    results_df = pd.merge(results_df, 
                       performance_df, 
                       left_index=True, 
                       right_index=True)
    
    results_df = results_df.T.drop_duplicates()
    results_df = results_df.T
    results_df = results_df.dropna()
    
    results_df.reset_index(drop=True, inplace=True)
    X = results_df.drop(['WT dice', 'TC dice', 'ET dice'], axis=1)  # Features
    y = results_df['WT dice']  # Target variable (Dice score)
    
    # Normalize features
    scaler = StandardScaler()
    _X = scaler.fit_transform(X)
    X = pd.DataFrame(_X, columns=X.columns)
    
    lr = GradientBoostingRegressor(random_state=42)
    
    efs = EFS(lr, 
              min_features=1,
              max_features=X.shape[1]-1,
              scoring='neg_mean_absolute_error',
              cv=5, 
              n_jobs=4, 
              print_progress=False)
    efs.fit(X, y)
    print('Best subset:', efs.best_feature_names_)
    return list(efs.best_feature_names_)
    
