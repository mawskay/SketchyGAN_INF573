import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from IPython.display import display 

import PIL
import os
from tqdm import tqdm
import cv2
import time
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import sys

from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm_notebook , tnrange

import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision import datasets
from tqdm.notebook import *
from torchcontrib.optim import SWA

import time
import matplotlib.pyplot as plt
import optuna

transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])
trans1= transforms.ToPILImage()

dataset_path = '/home/aous/Desktop/sketchyGAN/256x256/'
stats_path = '/home/aous/Desktop/sketchyGAN/info/stats.csv'
edges2shoes_path = '/home/aous/Desktop/sketchyGAN/pix2pix-pytorch/dataset/edges2shoes/train/'


class paths():
    def __init__(self, sketch = False, augmentation = 0):
        self.sketch = sketch
        self.augmentation = augmentation
    def get_path(self, ID):
        augments = ['tx_000000000000', 'tx_000100000000', 'tx_000000000010', 'tx_000000000110',
                    'tx_000000001010', 'tx_000000001110']
        path = dataset_path
        if self.sketch:
            path += 'sketch/' + augments[self.augmentation] + '/' + ID.split(',')[0].replace(' ', '_') + '/' + ID.split(',')[1] + '-' + ID.split(',')[2] + '.png'
        else:
            path += 'photo/' + augments[self.augmentation] + '/' + ID.split(',')[0].replace(' ', '_') + '/' + ID.split(',')[1] + '.jpg'
        return(path)

def get_spath(df, sketch, augmentation = 0):
    df['path'] = df['Category'] + ',' + df['ImageNetID'] + ',' + df['SketchID'].astype('str')
    func = paths(sketch = sketch, augmentation = augmentation)
    return(df.path.apply(func.get_path))

def get_dataframe(test_size, dataset_size, category = ''):
    df = pd.read_csv(stats_path)[['CategoryID', 'Category', 'ImageNetID', 'SketchID']]
    df['photo0_path'] = get_spath(df, False, 0)
    df['photo1_path'] = get_spath(df, False, 1)
    df['sketch0_path'] = get_spath(df, True, 0)
    df['sketch1_path'] = get_spath(df, True, 1)
    df['sketch2_path'] = get_spath(df, True, 2)
    df['sketch3_path'] = get_spath(df, True, 3)
    df['sketch4_path'] = get_spath(df, True, 4)
    df['sketch5_path'] = get_spath(df, True, 5)
    df = df.drop(['path'], axis = 1)
    if category != '':
        if len(df[df.Category == category]) == 0:
            raise Exception('none existing class')
        return(df[df.Category == category][:test_size], df[df.Category == category][test_size: test_size + dataset_size])
    return(df[:test_size], df[test_size: test_size + dataset_size])

class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, Table, augmented = False, randomize = True, noise_variance = 0.01):
        self.Table = Table
        self.noise_var = noise_variance
        self.randomizer = list(Table.index)
        self.augmented = augmented
        if randomize:
            np.random.shuffle(self.randomizer)
        self.transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        self.transform_rand = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x + self.noise_var*(torch.rand(x.shape)-0.5)),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    def __getitem__(self, index):
        photo_path = 'photo' + str(int(np.random.random()*2-1e-6)) + '_path'
        sketch_path = 'sketch' + str(int(np.random.random()*6-1e-6)) + '_path'
        photo = Image.open(self.Table[photo_path][self.randomizer[index]]).convert('RGB')
        sketch = Image.open(self.Table[sketch_path][self.randomizer[index]]).convert('RGB')
        
        if self.augmented:
            rot = (np.random.random()*2-1)*10
            if np.random.random() >= 0.5:
                photo = photo.transpose(Image.FLIP_LEFT_RIGHT)
                sketch = sketch.transpose(Image.FLIP_LEFT_RIGHT)
            converter = PIL.ImageEnhance.Color(photo)
            photo = converter.enhance((1.4-0.6)*np.random.random()+0.6)
            converter = PIL.ImageEnhance.Brightness(photo)
            photo = converter.enhance((1.4-0.6)*np.random.random()+0.6)
            photo = photo.rotate(rot)
            sketch = sketch.rotate(rot)
            
        x = self.transform_rand(sketch)
        label = self.transform(photo)
        return([x, label])
    def __len__(self):
        return(len(self.randomizer))

def crop(im, height, width):
    L = []
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            L.append(a)
    return(L)

class Custom_dataset_shoes(torch.utils.data.Dataset):
    def __init__(self, Range1, Range2, augmented = True, randomize = True, noise_variance = 0.005):
        self.noise_var = noise_variance
        self.randomizer = list(range(Range1, Range2))
        self.augmented = augmented
        if randomize:
            np.random.shuffle(self.randomizer)
        self.transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        self.transform_rand = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x + self.noise_var*(torch.rand(x.shape)-0.5)),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    def __getitem__(self, index):
        photo_path = edges2shoes_path + str(self.randomizer[index]) + '_AB.jpg'
        im = Image.open(photo_path).convert('RGB')
        sketch, photo = crop(im, int(im.size[1]), int(im.size[0]/2))
        
        if self.augmented:
            rot = (np.random.random()*2-1)*10
            if np.random.random() >= 0.5:
                photo = photo.transpose(Image.FLIP_LEFT_RIGHT)
                sketch = sketch.transpose(Image.FLIP_LEFT_RIGHT)
            converter = PIL.ImageEnhance.Color(photo)
            photo = converter.enhance((1.4-0.6)*np.random.random()+0.6)
            converter = PIL.ImageEnhance.Brightness(photo)
            photo = converter.enhance((1.4-0.6)*np.random.random()+0.6)
            photo = photo.rotate(rot)
            sketch = sketch.rotate(rot)
            
        x = self.transform_rand(sketch)
        label = self.transform(photo)
        return([x, label])
    def __len__(self):
        return(len(self.randomizer))