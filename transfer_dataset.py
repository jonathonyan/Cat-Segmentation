import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pdb

EXTRA_DATASET_PATH="./extra_data"
IMAGES = "images/images"
TRIMAPS = "annotations/annotations/trimaps"

class TransferDataset(Dataset):
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path if dataset_path else EXTRA_DATASET_PATH
        images, masks, datasize = self.get_transfer_dataset()
        self.X = torch.from_numpy(images).float()
        self.Y = torch.from_numpy(masks).float()
        self.datasize = datasize

    def __len__(self):
        return self.datasize


    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


    def get_transfer_dataset(self):
        folder_path = self.dataset_path
        inputs, masks = [], []
        input_path = join(folder_path, IMAGES)
        trimap_path = join(folder_path, TRIMAPS)

        input_filenames = listdir(input_path)

        datasize = 0

        for input_filename in input_filenames:
            if not input_filename.lower().endswith(".jpg"):
                continue

            image_name = input_filename.split(".")[0]

            image_filename = join(input_path, input_filename)
            trimap_filename = join(trimap_path, "{}.png".format(image_name))

            image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            trimap = cv2.imread(trimap_filename, cv2.IMREAD_GRAYSCALE)

            if type(image) == type(None) or type(trimap) == type(None):
                continue

            datasize += 1


            mask = np.zeros((image.shape[0], image.shape[1]))

            mask[np.where(trimap == 1)] = 255

            # cv2.imwrite("image_3.jpg", mask)
            # cv2.imwrite("mask_3.jpg", mask)

            # pdb.set_trace()

            image = cv2.resize(image, (128, 128))
            mask = cv2.resize(mask, (128, 128))


            image = np.transpose(image, (2, 0, 1))
            mask = mask.reshape(1, 128, 128)

            mask = (mask != 0)


            inputs.append(image)
            masks.append(mask)


        return np.array(inputs), np.array(masks), datasize

