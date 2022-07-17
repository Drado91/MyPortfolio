import zipfile
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os, random
import keras
from  torch.utils.data import DataLoader

main_path=os.getcwd()
zip_file_path = '{}/archive.zip'.format(main_path)
target_dir = '{}/dataset'.format(main_path)
print(zip_file_path)

def print_dir_hir(path):
    for dirpath, dirnames, filenames in os.walk(path):
        directory_level = dirpath.replace(path, "")
        directory_level = directory_level.count(os.sep)
        indent = " " * 4
        print("{}{}/".format(indent*directory_level, os.path.basename(dirpath)))

dataset_path=main_path+'/dataset'

dataset_group_path_train=dataset_path+'/Dataset_BUSI_with_GT/train'
normal_path_train = dataset_group_path_train+'normal/'
benign_path_train = dataset_group_path_train+'benign/'
malignant_path_train = dataset_group_path_train+'malignant/'

dataset_group_path_test=dataset_path+'/Dataset_BUSI_with_GT/test'
normal_path_test = dataset_group_path_test+'normal/'
benign_path_test = dataset_group_path_test+'benign/'
malignant_path_test = dataset_group_path_test+'malignant/'


def remove_masks(path):
    l=os.listdir(path)
    li=[]
    for f in l:
        if 'mask' not in f:
            li.append(f)
    return li
def delete_masks(path):
    l=os.listdir(path)
    for f in l:
        if 'mask' in f:
           os.remove(path+f)

train=DataLoader(dataset_group_path_train, batch_size=64, shuffle=True)
test=DataLoader(dataset_group_path_test, batch_size=64, shuffle=True)

print('finish')
"""
file_pos=random.choice(os.listdir(normal_path))
img_pos = cv2.imread(glacuma_pos_path+file_pos, cv2.IMREAD_COLOR)

glacuma_neg_path=train_path+'/Glaucoma_Negative/'
file_neg=random.choice(os.listdir(glacuma_neg_path))
img_neg = cv2.imread(glacuma_neg_path+file_neg, cv2.IMREAD_COLOR)

plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.imshow(img_pos,aspect='auto')
plt.title("Glacuma Positive")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_neg,aspect='auto')
plt.title("Glacuma Negative")
plt.axis('off')
plt.show()

print(f' image shape is {img_pos.shape}')
"""