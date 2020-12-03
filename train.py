import argparse

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

from dataset import *
from models import *

transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])
trans1= transforms.ToPILImage()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def objective(trial):

    # Generate the models.
    generator = GeneratorUNet()
    generator.load_state_dict(torch.load('generator_edges_shoes.pth'))
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load('discriminator_edges_shoes.pth'))
    #generator.apply(weights_init_normal)
    #discriminator.apply(weights_init_normal)

    n_epoch = int(trial.suggest_discrete_uniform("epochs", 30, 100, 10))
    # Generate the optimizers.
    optimizer_name_G = trial.suggest_categorical("optimizer_G", ["Adam", "RMSprop"])
    lr_G = trial.suggest_loguniform("lr_G", 1e-5, 1e-3)
    optimizer_G = getattr(optim, optimizer_name_G)(generator.parameters(), lr=lr_G)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max = n_epoch + 10, eta_min=0, last_epoch=-1)
    
    optimizer_name_D = trial.suggest_categorical("optimizer_D", ["Adam", "RMSprop"])
    lr_D = trial.suggest_loguniform("lr_D", 1e-5, 1e-3)
    optimizer_D = getattr(optim, optimizer_name_D)(discriminator.parameters(), lr=lr_D)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max = n_epoch + 10, eta_min=0, last_epoch=-1)
    
    test_ds = Custom_dataset(df[df.Category == 'shoe'][:6], augmented = False, noise_variance = 0.01)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size = 6, pin_memory = True)
    train_ds = Custom_dataset(df[df.Category == 'shoe'][6:], augmented = False, noise_variance = trial.suggest_loguniform("noise", 0.001, 0.5))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = 64, num_workers=4, pin_memory = True)
    
    cuda = True
    train_acc_period = 5
    lambda_pixel = 100
    visualize = True
                                                 
    loss_train = []
    loss_test = []
    total = 0
    
    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        #net, optimizer = amp.initialize(net, optimizer, opt_level = 'O1')
    for epoch in tnrange(n_epoch):  # loop over the dataset multiple times
        running_loss_G = 0.0
        running_loss_D = 0.0
        print("epoch [", epoch, "/", n_epoch, "]")
        print("generator s lr: ", scheduler_G.get_last_lr())
        print("discriminator s lr: ",scheduler_D.get_last_lr())
        criterion_GAN = torch.nn.MSELoss()
        criterion_pixelwise = torch.nn.L1Loss()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            # get the inputs
            if cuda:
                #inputs = inputs.type(torch.cuda.HalfTensor)
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)
                patch = (1, inputs.size(2) // 2 ** 4, inputs.size(3) // 2 ** 4)
                valid = Variable(torch.Tensor(np.ones((labels.size(0), *patch))), requires_grad=False).cuda()
                fake = Variable(torch.Tensor(np.zeros((labels.size(0), *patch))), requires_grad=False).cuda()
                
            # ------------------
            #  Train Generator
            # ------------------

            optimizer_G.zero_grad()
        
            output_gen = generator(inputs)
            output_dis_fake = discriminator(output_gen, inputs)
            # GAN loss
            loss_GAN = criterion_GAN(output_dis_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(output_gen, labels)
            
            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()
            
            # ------------------
            #  Train Discriminator
            # ------------------
            
            # zero the parameter gradients
            optimizer_D.zero_grad()
            output_dis_real = discriminator(labels, inputs)
            # Real loss
            loss_real = criterion_GAN(output_dis_real, valid)

            # Fake loss
            output_dis_fake = discriminator(output_gen.detach(), inputs)
            loss_fake = criterion_GAN(output_dis_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()
            
            total += labels.size(0)
            # print statistics
            running_loss_G = 0.33*loss_G.item()/labels.size(0) + 0.66*running_loss_G
            running_loss_D = 0.33*loss_D.item()/labels.size(0) + 0.66*running_loss_D
            loss_train.append(running_loss_G)
            if visualize:
                if i % train_acc_period == train_acc_period-1:
                    print('[%d, %5d] loss_G: %.3f' %(epoch + 1, i + 1, running_loss_G))
                    print('[%d, %5d] loss_D: %.3f' %(epoch + 1, i + 1, running_loss_D))
                    running_loss_G = 0.0
                    running_loss_D = 0.0
                    inputs, labels = next(iter(test_loader))
                    test_images = generator(inputs.cuda())
                    im = trans1(test_images.cpu()[0]).resize((128,128), Image.BILINEAR)
                    im2 = trans1(labels.cpu()[0]).resize((128,128), Image.BILINEAR)
                    for i in range(test_images.size(0)-1):
                        temp = trans1(test_images.cpu()[i+1]).resize((128,128), Image.BILINEAR)
                        temp2 = trans1(labels.cpu()[i+1]).resize((128,128), Image.BILINEAR)
                        im = get_concat_h(im, temp)
                        im2 = get_concat_h(im2, temp2)
                    display(im)
                    display(im2)
            inputs = None
            output_dis_real = None
            output_dis_fake = None
            output_gen = None
            labels = None
        scheduler_G.step()
        scheduler_D.step()
        
        inputs, labels = next(iter(test_loader))
        inputs = inputs.cuda()
        labels = labels.cuda()
        valid = Variable(torch.Tensor(np.ones((labels.size(0), *patch))), requires_grad=False).cuda()
        fake = Variable(torch.Tensor(np.zeros((labels.size(0), *patch))), requires_grad=False).cuda()
        output_gen = generator(inputs)
        output_dis_fake = discriminator(output_gen, inputs)
        # GAN loss
        loss_GAN = criterion_GAN(output_dis_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(output_gen, labels)

        # Total loss
        test_loss_G = loss_GAN + lambda_pixel * loss_pixel
        trial.report(test_loss_G, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
      
    if visualize:
        print('Finished Training')
    
    accuracy = test_loss_G
    return accuracy

def engine(opt, test_loader, train_loader):
    cuda = True
    train_acc_period = 5
    lambda_pixel = 100
    visualize = True

    # Generate the models.
    generator = GeneratorUNet()
    generator.load_state_dict(torch.load(opt.weights_g))
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(opt.weights_d))

    n_epoch = opt.n_epochs
    # Generate the optimizers.
    lr_G = opt.lr_G
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr_G)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max = n_epoch + 10, eta_min=0, last_epoch=-1)

    lr_D = opt.lr_D
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr_D)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max = n_epoch + 10, eta_min=0, last_epoch=-1)

    loss_train = []
    loss_test = []
    total = 0

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        #net, optimizer = amp.initialize(net, optimizer, opt_level = 'O1')
    for epoch in range(opt.n_epochs):  # loop over the dataset multiple times
        running_loss_G = 0.0
        running_loss_D = 0.0
        print("epoch [", epoch, "/", n_epoch, "]")
        print("generator s lr: ", scheduler_G.get_last_lr())
        print("discriminator s lr: ",scheduler_D.get_last_lr())
        criterion_GAN = torch.nn.MSELoss()
        criterion_pixelwise = torch.nn.L1Loss()
        for i, (inputs, labels) in enumerate(train_loader):
            # get the inputs
            if cuda:
                #inputs = inputs.type(torch.cuda.HalfTensor)
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)
                patch = (1, inputs.size(2) // 2 ** 4, inputs.size(3) // 2 ** 4)
                valid = Variable(torch.Tensor(np.ones((labels.size(0), *patch))), requires_grad=False).cuda()
                fake = Variable(torch.Tensor(np.zeros((labels.size(0), *patch))), requires_grad=False).cuda()

            # ------------------
            #  Train Generator
            # ------------------

            optimizer_G.zero_grad()

            output_gen = generator(inputs)
            output_dis_fake = discriminator(output_gen, inputs)
            # GAN loss
            loss_GAN = criterion_GAN(output_dis_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(output_gen, labels)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ------------------
            #  Train Discriminator
            # ------------------

            # zero the parameter gradients
            optimizer_D.zero_grad()
            output_dis_real = discriminator(labels, inputs)
            # Real loss
            loss_real = criterion_GAN(output_dis_real, valid)

            # Fake loss
            output_dis_fake = discriminator(output_gen.detach(), inputs)
            loss_fake = criterion_GAN(output_dis_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            total += labels.size(0)
            # print statistics
            running_loss_G = 0.33*loss_G.item()/labels.size(0) + 0.66*running_loss_G
            running_loss_D = 0.33*loss_D.item()/labels.size(0) + 0.66*running_loss_D
            loss_train.append(running_loss_G)
            if visualize:
                if i % train_acc_period == train_acc_period-1:
                    print('[%d, %5d] loss_G: %.3f' %(epoch + 1, i + 1, running_loss_G))
                    print('[%d, %5d] loss_D: %.3f' %(epoch + 1, i + 1, running_loss_D))
                    running_loss_G = 0.0
                    running_loss_D = 0.0
                    inputs, labels = next(iter(test_loader))
                    test_images = generator(inputs.cuda())
                    im = trans1(test_images.cpu()[0]).resize((128,128), Image.BILINEAR)
                    im2 = trans1(labels.cpu()[0]).resize((128,128), Image.BILINEAR)
                    for i in range(test_images.size(0)-1):
                        temp = trans1(test_images.cpu()[i+1]).resize((128,128), Image.BILINEAR)
                        temp2 = trans1(labels.cpu()[i+1]).resize((128,128), Image.BILINEAR)
                        im = get_concat_h(im, temp)
                        im2 = get_concat_h(im2, temp2)
                    im.save(opt.test_img_path+"_fake.jpg")
                    im2.save(opt.test_img_path+"_real.jpg")
            inputs = None
            output_dis_real = None
            output_dis_fake = None
            output_gen = None
            labels = None
        scheduler_G.step()
        scheduler_D.step()

        inputs, labels = next(iter(test_loader))
        inputs = inputs.cuda()
        labels = labels.cuda()
        valid = Variable(torch.Tensor(np.ones((labels.size(0), *patch))), requires_grad=False).cuda()
        fake = Variable(torch.Tensor(np.zeros((labels.size(0), *patch))), requires_grad=False).cuda()
        output_gen = generator(inputs)
        output_dis_fake = discriminator(output_gen, inputs)
        # GAN loss
        loss_GAN = criterion_GAN(output_dis_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(output_gen, labels)

        # Total loss
        test_loss_G = loss_GAN + lambda_pixel * loss_pixel
        print(test_loss_G.item())

        # Handle pruning based on the intermediate value.
        

    if visualize:
        print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_d', type=str, default='discriminator_edges_shoes.pth', help='weights of discriminator')
    parser.add_argument('--weights_g', type=str, default='generator_edges_shoes.pth', help='weights of generator')
    parser.add_argument('--optuna', type=bool, default=False, help='hyperparameter search with optima')
    parser.add_argument('--trials', type=int, default=30, help='number of trial for hyperparameter search with optima')
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr_G', type=float, default=5e-4, help='generator s learning rate')
    parser.add_argument('--lr_D', type=float, default=2e-4, help='discriminator s learning rate')
    parser.add_argument('--noise', type=int, default=0.1, help='noise variance')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=6, help='images shown for validation')
    parser.add_argument('--test_img_path', type=str, default='./test_images', help='path of images shown for validation')
    parser.add_argument('--edges2shoes', type=bool, default=True, help='dataset type')
    parser.add_argument('--category', type=bool, default='shoes', help='class to train')
    parser.add_argument('--dataset_size', type=int, default=40000, help='number of images processed')
    opt = parser.parse_args()
    
    if opt.optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=opt.trials)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        if opt.edges2shoes:
            test_ds = Custom_dataset_shoes(1, opt.test_batch_size, augmented = False, noise_variance = opt.noise)
            train_ds = Custom_dataset_shoes(7, opt.test_batch_size + opt.dataset_size, augmented = True, noise_variance = opt.noise)
        else:
            test_df , train_df = get_dataframe(opt.test_batch_size, opt.dataset_size, 'shoe')
            test_ds = Custom_dataset(test_df, augmented = False, noise_variance = opt.noise)
            train_ds = Custom_dataset(train_df, augmented = False, noise_variance = opt.noise)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size = opt.test_batch_size, pin_memory = True)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size = opt.batch_size, num_workers=4, pin_memory = True)

        engine(opt, test_loader, train_loader)
