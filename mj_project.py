import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# MJ imports
from segdataset import SegmentationDataset
import mj_fcn_resnet101 
import mj_trainer
import torchvision.models.segmentation
import torchvision.models
#import matplotlib.patches as mpatches
#import mj_utils

# traininng 
seg_dataset_train = SegmentationDataset("mj_data_set/Train", "Images", "Masks", transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
seg_dataloader_train = DataLoader(seg_dataset_train, batch_size=20, shuffle=True)
seg_dataset_test = SegmentationDataset("mj_data_set/Test", "Images", "Masks", transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
seg_dataloader_test = DataLoader(seg_dataset_test, batch_size=20, shuffle=True)
model = mj_fcn_resnet101.createFCNResnet101(26,feature_extract=True)
# If continuation of training needed
# model.load_state_dict(torch.load("modely/15.pt"))
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = mj_trainer.train_model(model,criterion, data_loaders, optimizer,{},".",25,0)
#print(model)

