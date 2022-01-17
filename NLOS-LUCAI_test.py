from torchvision.transforms import transforms
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from fully_connected import FC
from localization_error import LocalizationError
from localization_loss import LocalizationLoss
from BICN import BICN
import torch
import torch.nn.functional as F
import torch.nn as nn
import Unet
import time
import math
import copy
import math
import os
import numpy as np
import matplotlib.pyplot as plt


class MyDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.names_list[idx].split('\t')[0])
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = Image.open(image_path) 
        label_x = float(self.names_list[idx].split('\t')[1])
        label_y = float(self.names_list[idx].split('\t')[2])
        if self.transform:
            image = self.transform(image)

        return image, label_x, label_y


data_transforms = {
    'test': transforms.Compose([
        transforms.ToTensor(),
    ])
}


test_dataset = MyDataset(root_dir=r'E:\Data\test',
                         names_file=r'E:\Data\test_label.txt',
                         transform=data_transforms['test'])


dataloaders = {'test': torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, num_workers=2)}
dataset_sizes = {'test': len(test_dataset)}


def test_model(net1, net2, error):
    running_error_x = 0.0
    running_error_y = 0.0
    error_sum = 0.0
    squared_error_sum = 0.0
    pre_x = []
    pre_y = []
    dist = []

    for data in dataloaders['test']:
        since = time.time()

        inputs, label_x, label_y = data
        label_x = torch.Tensor(label_x.float())
        label_y = torch.Tensor(label_y.float())

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            label_x = label_x.cuda()
            label_y = label_y.cuda()

        output_img = net1(inputs)
        outputs = net2(output_img)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))

        pre_x.append(outputs.cuda().data.cpu().numpy()[0][0])
        pre_y.append(outputs.cuda().data.cpu().numpy()[0][1])

        error_x, error_y = error(outputs, label_x, label_y)

        error_sum += math.sqrt(pow(error_x.item(), 2) + pow(error_y.item(), 2))
        error_dist = math.sqrt(pow(error_x.item(), 2) + pow(error_y.item(), 2))
        dist.append(error_dist)

        running_error_x += error_x.item() * inputs.size(0)
        running_error_y += error_y.item() * inputs.size(0)
        squared_error_sum += pow(error_x.item(), 2) + pow(error_y.item(), 2)
    
    mean_error = error_sum / dataset_sizes['test']

    mean_error_x = running_error_x / dataset_sizes['test']
    mean_error_y = running_error_y / dataset_sizes['test']

    root_mean_squared_error = math.sqrt(squared_error_sum / dataset_sizes['test'])

    return pre_x, pre_y, dist, mean_error_x, mean_error_y, mean_error, root_mean_squared_error


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net1 = BICN(3,3).to(device=device)

    net2 = models.resnet18()
    net2.fc = FC()
    net2.to(device=device)

    net1.load_state_dict(torch.load(r'E:\model\net1_model.pkl', map_location=device))
    net1.eval()

    net2.load_state_dict(torch.load(r'E:\model\net2_model.pkl', map_location=device))
    net2.eval()

    error = LocalizationError()

    pre_x, pre_y, dist, mean_error_x, mean_error_y, mean_error, root_mean_squared_error = test_model(net1, net2, error)

    MAE = mean_error
    RMSE = root_mean_squared_error

    print('mean_error_x:{:.4f}'.format(mean_error_x), 'mean_error_y:{:.4f}'.format(mean_error_y))
    print('MAE:{:.8f}'.format(MAE), 'RMSE:{:.8f}'.format(RMSE))