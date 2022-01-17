from torch.utils.data import Dataset
from PIL import Image
import os

# define the class MyDataset
class MyDataset(Dataset):
    def __init__(self, root1, root2, names_file, transform = None, target_transform = None):
        n = len(os.listdir(root1))
        
        imgs = []
        for i in range(n):
            img = os.path.join(root1, "%05d.bmp"%i)
            label = os.path.join(root2, "%05d.bmp"%i)
            imgs.append([img,label])
        
        self.root_dir = root1
        self.imgs = imgs
        self.names_file = names_file
        self.transform = transform
        self.target_transform = target_transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1
    
    
    def __getitem__(self,index):
        x_path,y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        image_path = os.path.join(self.root_dir, self.names_list[index].split('\t')[0])
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = Image.open(image_path) 
        label_x = float(self.names_list[index].split('\t')[1])
        label_y = float(self.names_list[index].split('\t')[2])
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x,img_y,label_x,label_y
    
    
    def __len__(self):
        return len(self.imgs)