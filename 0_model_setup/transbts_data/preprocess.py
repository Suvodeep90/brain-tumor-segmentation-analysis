import pickle
import os
import numpy as np
import nibabel as nib
import pandas as pd

# test_set = {
#         'root': 'path to testing set',
#         'flist': 'test.txt',
#         'has_label': False
#         }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = np.asarray(proxy.dataobj)
    # print(data)
    return data


def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = path + 'data_i16.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return
    
def convert_labels_from_BraTS2023_to_BraTS2020(seg):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 0] = 0
    new_seg[seg == 1] = 1
    new_seg[seg == 2] = 2
    new_seg[seg == 3] = 4
    return new_seg


def process_f32b0(path, save_path, name, partition, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    # if has_label:
    #     label =nib_load(path + '/' + name  + '_seg.nii')
    if partition == 'test':
        modalities = ('t2f', 't1c', 't1n', 't2w')
        if has_label:
            label =nib_load(path + '/' + name  + '-seg.nii.gz')
        label = convert_labels_from_BraTS2023_to_BraTS2020(label)
    else:
        modalities = ('flair', 't1ce', 't1', 't2')
        label =nib_load(path + '/' + name  + '_seg.nii')
    images = np.stack([nib_load(path + '/' + name  + '-' + modal + '.nii.gz') for modal in modalities], -1)  # [240,240,155]
    output = save_path + '/' + name +'_data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(4):
        x = images[..., k]  #
        x = x.astype(np.float64)
        y = x[mask]
        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


# def doit(dset):
#     root, has_label = dset['root'], dset['has_label']
#     file_list = os.path.join(root, dset['flist'])
#     subjects = open(file_list).read().splitlines()
#     names = [sub.split('/')[-1] for sub in subjects]
#     paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

#     for path in paths:

#         process_f32b0(path, has_label)


def doit(dset):
    root, has_label, save_path, partition = dset['root'], dset['has_label'], dset['save_path'], dset['partition']
    file_list = pd.read_csv(root + dset['flist'])
    paths = file_list.path.values.tolist()
    names = file_list.Brats20ID.values.tolist()
    # paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

    for i in range(len(paths)):
        path = paths[i]
        name = names[i]
        process_f32b0(path, save_path, name, partition, has_label)


if __name__ == '__main__':

    # # train
    # train_set = {
    #         'root': '/proj/arise/arise/suvodeep/Data/BraTs2020/Processed_data/',
    #         'save_path': '/proj/arise/arise/suvodeep/Data/TransBTS/Train',
    #         'flist': 'train_df.csv',
    #         'has_label': True,
    #         'partition': 'train'
    #         }

    # # test/validation data
    # valid_set = {
    #         'root': '/proj/arise/arise/suvodeep/Data/BraTs2020/Processed_data/',
    #         'save_path': '/proj/arise/arise/suvodeep/Data/TransBTS/Val',
    #         'flist': 'val_df.csv',
    #         'has_label': True,
    #         'partition': 'valid'
    #         }

    # test_set = {
    #         'root': '/proj/arise/arise/suvodeep/Data/BraTs2020/Processed_data/',
    #         'save_path': '/proj/arise/arise/suvodeep/Data/TransBTS/Test',
    #         'flist': 'test_df.csv',
    #         'has_label': True,
    #         'partition': 'test'
    #         }
    
    PED_test_set = {
            'root': '/proj/arise/arise/suvodeep/Data/BraTs2023/BraTS-MEN/',
            'save_path': '/proj/arise/arise/suvodeep/Data/TransBTS/MEN_Test',
            'flist': 'train_df.csv',
            'has_label': True,
            'partition': 'test'
            }
    
    # doit(train_set)
    # doit(valid_set)
    doit(PED_test_set)

