from torchvision.transforms import transforms
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from my_dataset import MyDataset
from fully_connected import FC
from localization_error import LocalizationError
from localization_loss import LocalizationLoss
from BICN import BICN
import torch
import torch.nn as nn
import time
import copy
import math
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


#data transform
data_transforms = {
    'train': transforms.Compose([transforms.ToTensor()]),
    'test': transforms.Compose([transforms.ToTensor()]),
}

train_dataset = MyDataset(root1=r'E:\Data\train',
                          root2=r'E:\Data\train_label',
                          names_file=r'E:\Data\train_label.txt',
                          transform=data_transforms['train'],
                          target_transform=data_transforms['train'])

test_dataset =  MyDataset(root1=r'E:\Data\test',
                          root2=r'E:\Data\test_label',
                          names_file=r'E:\Data\test_label.txt',
                          transform=data_transforms['test'],
                          target_transform=data_transforms['test'])

dataloaders = {'train': torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=8, num_workers=2),
               'test': torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=8, num_workers=2)}


dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}

# define the train and validation function 
# each epoch validation follows the training
def train(net1, net2, criterion1, criterion2, error, optimizer1, optimizer2, scheduler1, scheduler2, num_epochs):
    since = time.time()
    best_model_wts1 = copy.deepcopy(net1.state_dict())
    best_model_wts2 = copy.deepcopy(net2.state_dict())
    mini_loss = float("inf")
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
            
        tol_loss1 = 0.0
        tol_loss2 = 0.0
        running_loss = 0.0
        running_error_x = 0.0
        running_error_y = 0.0
        net1.train(True)
        net2.train(True)

        for data in dataloaders['train']:

            img_x, img_y, label_x, label_y = data
            label_x = torch.Tensor(label_x.float())
            label_y = torch.Tensor(label_y.float())

            if torch.cuda.is_available():
                img_x = img_x.cuda()
                img_y = img_y.cuda()
                label_x = label_x.cuda()
                label_y = label_y.cuda()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            output_img = net1(img_x)

            loss1 = criterion1(output_img, img_y)

            outputs = net2(output_img) 

            loss2 = criterion2(outputs, label_x, label_y)

            loss = loss1 + 20*loss2

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            with torch.no_grad():
                error_x, error_y = error(outputs, label_x, label_y) 
                tol_loss1 += loss1.item() * img_x.size(0)
                tol_loss2 += loss2.item() * img_x.size(0)
                running_loss += loss.item() * img_x.size(0)
                running_error_x += error_x.item() * img_x.size(0)
                running_error_y += error_y.item() * img_x.size(0)

        with torch.no_grad():
            epoch_loss1 = tol_loss1 / dataset_sizes['train']
            epoch_loss2 = tol_loss2 / dataset_sizes['train']
            epoch_loss = running_loss / dataset_sizes['train']
            epoch_error_x = running_error_x / dataset_sizes['train']
            epoch_error_y = running_error_y / dataset_sizes['train']

        scheduler1.step()
        scheduler2.step()

        print('{} Loss1: {:.8f} Loss2: {:.8f} Loss: {:.8f} Error_x: {:.4f} Error_y：{:.4f}'.format(
            phase, epoch_loss1, epoch_loss2, epoch_loss, epoch_error_x, epoch_error_y))

        if epoch_loss2 < mini_loss:
            mini_loss = epoch_loss2
            loss_epoch = epoch
            best_model_wts1 = copy.deepcopy(net1.state_dict())
            best_model_wts2 = copy.deepcopy(net2.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimum test loss: {:.8f}'.format(mini_loss))

    net1.load_state_dict(best_model_wts1)
    net2.load_state_dict(best_model_wts2)
    return net1, net2


if __name__ == '__main__':

    epoch = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net1 = BICN(3,3).to(device)
    net2 = models.resnet18(pretrained=True)
    net2.fc = FC()

    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()

    criterion1 = torch.nn.SmoothL1Loss()
    criterion2 = LocalizationLoss()

    error = LocalizationError()

    optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.00015, betas=(0.9,0.999))
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.0001,  betas=(0.9,0.999))

    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=5,  gamma=0.7)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=1,  gamma=0.9)

    net1_model, net2_model = train(net1, net2, criterion1, criterion2, error, optimizer1, optimizer2, exp_lr_scheduler1, exp_lr_scheduler2, num_epochs=epoch)
    
    torch.save(net1_model.state_dict(),r"E:\model\net1_model.pkl")
    torch.save(net2_model.state_dict(),r"E:\model\net2_model.pkl")
