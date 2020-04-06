import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from augmentation import *


import pdb



TRAIN_DATASET_PATH="./cat_data/Train"
TEST_DATASET_PATH="./cat_data/Test"

class CatDataset(Dataset):
    def __init__(self, train=True, with_augmentation=False):
        cats, masks, datasize = self.get_cat_dataset(train, with_augmentation)
        self.X = torch.from_numpy(cats).float()
        self.Y = torch.from_numpy(masks).float()
        self.datasize = datasize


    def __len__(self):
        return self.datasize

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


    def get_cat_dataset(self, train=True, with_augmentation=False):

        folder_path = TRAIN_DATASET_PATH if train else TEST_DATASET_PATH
        inputs, masks = [], []
        input_path = join(folder_path, "input")
        mask_path = join(folder_path, "mask")

        input_filenames = listdir(input_path)

        datasize = 0

        for input_filename in input_filenames:
            if not input_filename.lower().endswith(".jpg"):
                continue
            data_id = input_filename.split(".")[1]

            cat_filename = join(input_path, input_filename)
            mask_filename = join(mask_path, "mask_cat.{}.jpg".format(data_id))

            cat = cv2.imread(cat_filename, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)


            if type(cat)!=type(None) and type(mask)!=type(None):

                cats_each = [cat]
                masks_each = [mask]


                if with_augmentation:
                    function_idx = np.random.randint(4)
                    augmentation_fn = AUGMENTATION_FUNCTIONS[function_idx]
                    cat_augmented, mask_augmented = augmentation_fn(cat, mask)

                    cats_each.append(cat_augmented)
                    masks_each.append(mask_augmented)

                for cat, mask in zip(cats_each, masks_each):
                    try:
                        cat = cv2.resize(cat, (128, 128))
                        mask = cv2.resize(mask, (128, 128))
                    except Exception as e:
                        print(str(e))
                        continue
                    datasize+=1

                    cat = np.transpose(cat, (2, 0, 1))
                    mask = mask.reshape(1, 128, 128)

                    mask = (mask != 0)

                    inputs.append(cat)
                    masks.append(mask)

        return np.array(inputs), np.array(masks), datasize











