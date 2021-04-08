import json
import os
from PIL import Image
from torchvision import transforms
import random
import numpy as np

# def generate_samples(classLabels, samplesPerImage, inImgDir, inLabDir, outImgDir, outLabDir):
#     imCount = 0 
#     fileNames = os.listdir(inImgDir)
#     testPortion = 0.1
#     for fn in fileNames:
#         base_filename, filename_suffix = os.path.splitext(fn)
#         img = Image.open(os.path.join(inImgDir, base_filename  + filename_suffix))
#         img_label = Image.open(os.path.join(inLabDir, base_filename + filename_suffix))
#         imCount = 0
#         while imCount<samplesPerImage:
#             pars = transforms.RandomCrop.get_params(img,(224,224))
#             imsize = img.size
            
#             if pars[0]+pars[2] < imsize[0] and pars[1]+pars[3] < imsize[1]:
#                 train_test = "Test" if random.uniform(0,1) < testPortion else "Train"
#                 cropped_image = img.crop((pars[0], pars[1], pars[0]+pars[2], pars[1]+pars[3]))
#                 cropped_label_img = img_label.crop((pars[0], pars[1], pars[0]+pars[2], pars[1]+pars[3]))
#                 out_base_fname = f"{train_test}_{base_filename}_{pars[0]}_{pars[1]}"
#                 cropped_image.save(os.path.join(outImgDir, out_base_fname  + filename_suffix))
#                 cropped_label_img.save(os.path.join(outLabDir, out_base_fname + filename_suffix))
#                 cropped_image.close()
#                 cropped_label_img.close()
#                 imCount += 1
#         img_label.close()
#         img.close()

def generate_samples2(samplesPerImage, inImgDir, inLabDir, outRootDir, testPortion=0.1):
    imCount = 0 
    fileNames = os.listdir(inImgDir)
    outTrainDir = os.path.join(outRootDir, "Train")
    outTestDir =  os.path.join(outRootDir, "Test")
   
    for fn in fileNames:
        base_filename, filename_suffix = os.path.splitext(fn)
        img = Image.open(os.path.join(inImgDir, base_filename  + filename_suffix))
        img_label = Image.open(os.path.join(inLabDir, base_filename + filename_suffix))
        imCount = 0
        while imCount<samplesPerImage:
            pars = transforms.RandomCrop.get_params(img,(224,224))
            imsize = img.size
            
            if pars[0]+pars[2] < imsize[0] and pars[1]+pars[3] < imsize[1]:
                train_test = "Test" if random.uniform(0,1) < testPortion else "Train"
                cropped_image = img.crop((pars[0], pars[1], pars[0]+pars[2], pars[1]+pars[3]))
                cropped_label_img = img_label.crop((pars[0], pars[1], pars[0]+pars[2], pars[1]+pars[3]))
                out_base_fname = f"{train_test}_{base_filename}_{pars[0]}_{pars[1]}"
                cropped_image.save(os.path.join(outRootDir, train_test, "Images", out_base_fname  + filename_suffix))
                cropped_label_img.save(os.path.join(outRootDir, train_test, "Masks", out_base_fname + filename_suffix))
                cropped_image.close()
                cropped_label_img.close()
                imCount += 1
        img_label.close()
        img.close()


def get_label_maps():
    with open('./dataset/classes.json') as f:
        mapRGB2ClassId = {}
        mapClassId2Name = {}
        mapClassId2RGB = []

        jsclasses = json.load(f)
        for k in jsclasses:
            mapRGB2ClassId[k[2]] = k[0]
            mapClassId2Name[k[0]] = k[1]

            mapClassId2RGB.append(np.array((int(k[2][0:2], 16), int(k[2][2:4], 16),int(k[2][4:6], 16),)))
            
        return (mapClassId2Name,mapClassId2RGB)
