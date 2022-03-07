import os
from natsort import natsorted
import numpy as np
import pandas as pd
import json
import glob
from torch.utils.data import Dataset
import torch
import logging
from PIL import Image
from torchvision import transforms


class SurgeryVideoDataset(Dataset):
    def __init__(self,
                 tensor_dir
                 ):
        logging.basicConfig(level=logging.DEBUG)
        # tensor dir
        self.tensor_dir = tensor_dir
        self.tensors = natsorted(glob.glob(os.path.join(self.tensor_dir, '*pt')))


    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        sample = torch.load(self.tensors[idx])

        return sample

