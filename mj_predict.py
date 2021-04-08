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
#import json
import time

# MJ imports
from segdataset import SegmentationDataset
import mj_fcn_resnet101 
import torchvision.models.segmentation
import torchvision.models
import matplotlib.patches as mpatches
import mj_utils
#%matplotlib inline

# Define the helper function
def decode_segmap(image, mapClassId2RGB, nc=26 ):
    label_colors = mapClassId2RGB
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
  
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l][0]
        g[idx] = label_colors[l][1]
        b[idx] = label_colors[l][2]
    
    rgb = np.stack([r, g, b], axis=2)
    rgb = rgb.squeeze()
    rgb = np.transpose(rgb, (0, 2, 1))
    return (label_colors,rgb)


(mapClassId2Name, mapClassId2RGB) = mj_utils.get_label_maps()
classLabels = list(mapClassId2Name.values())

def get_mask_for_crop(im, model, c,r,ws, transform):
    cropped_image = im.crop((c,r,c+ws,r+ws))
    cropped_image = transform(cropped_image)
    cropped_image = cropped_image.unsqueeze(0)
    mj_output = model(cropped_image)
    return torch.argmax(mj_output["out"], dim=1)
    

def segment_image(model, filename, mapClassId2RGB, iterace):
    with Image.open(filename) as im:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        width, height = im.size
        winsize = 224
        result_mask = torch.tensor(torch.zeros([1,height,width], dtype=torch.int64))

        f, axarr = plt.subplots(1,2, figsize=(23,7))
        axarr[0].imshow(im)
        
        r = 0
        while r + winsize <= height:
            c = 0
            while c + winsize <= width:
                result_mask[0,r:r+winsize, c:c+winsize] = get_mask_for_crop(im, model, c,r, winsize, transform)
                c += winsize

            # width overflow
            if c < width:
                c = width - winsize
                result_mask[0,r:r+winsize, c:c+winsize] = get_mask_for_crop(im, model, c,r, winsize, transform)
            r += winsize

        if r < height:
            r = height - winsize
            c = 0
            while c + winsize <= width:
                result_mask[0,r:r+winsize, c:c+winsize] = get_mask_for_crop(im, model, c,r, winsize, transform)
                c += winsize

            # width overflow
            if c < width:
                c = width - winsize
                result_mask[0,r:r+winsize, c:c+winsize] = get_mask_for_crop(im, model, c,r, winsize, transform)           
        (label_colors, rgb) = decode_segmap(result_mask, mapClassId2RGB)

        axarr[1].imshow(rgb)
        num_classes = 26
        
        patches = [ mpatches.Patch(color=mapClassId2RGB[i]/255, label=f"{classLabels[i]}") for i in range(num_classes)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0. )
        #plt.show()
        fileNameOut = os.path.basename(filename)+"_"+str(iterace)+"_out.tif"
        plt.savefig(fileNameOut)


        
        

model = mj_fcn_resnet101.createFCNResnet101(26)

#for i in [24, 23, 22, 21, 20, 19, 18, 17, 15, 14, 13, 12, 10, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
for i in [23]:
    modelName = f"modely/{i}.pt"
    model.load_state_dict(torch.load(modelName))
    print(f"MJ model_name:{modelName}")
    model.eval()
    filelist=os.listdir('dataset\\rgb')
    for fichier in filelist:
        if fichier.endswith(".tif"):
            segment_image(model, "dataset\\rgb\\"+fichier, mapClassId2RGB, i)

