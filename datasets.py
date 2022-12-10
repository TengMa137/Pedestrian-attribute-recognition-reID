from cProfile import label
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
#from skimage import io
from PIL import Image

#mydataset
class ImagesData(Dataset):
    def __init__(self, csv_path, img_path, transform = None):
        """
        Args:
            csv_path (string): csv file path
            img_path (string): image path
            transform: transform 
        """
        self.file_path = img_path
        self.transform = transform

        self.annotations = pd.read_csv(csv_path, sep=',') # annotations csv file, header=None
        self.annotations['upmulticolor'] = np.ones(751).astype(int)
        self.annotations['downmulticolor'] = np.ones(751).astype(int)
        self.annotations['young'] = np.ones(751).astype(int)
        self.annotations['teenager'] = np.ones(751).astype(int)
        self.annotations['adult'] = np.ones(751).astype(int)
        self.annotations['old'] = np.ones(751).astype(int)

        self.annotations['upmulticolor'] = self.annotations['upmulticolor'].where(self.annotations.iloc[:,11:19].sum(axis=1) > 8, 2)
        self.annotations['downmulticolor'] = self.annotations['downmulticolor'].where(self.annotations.iloc[:, 19:28].sum(axis=1) > 9, 2)
        self.annotations['young'] = self.annotations['young'].where(self.annotations.iloc[:,1] != 1, 2) 
        self.annotations['teenager'] = self.annotations['teenager'].where(self.annotations.iloc[:,1] != 2, 2)
        self.annotations['adult'] = self.annotations['adult'].where(self.annotations.iloc[:,1] != 3, 2)
        self.annotations['old'] = self.annotations['old'].where(self.annotations.iloc[:,1] != 4, 2)
        self.annotations['id'] = self.annotations['id'] + 1
        self.annotations =self.annotations - 1
        self.annotations = self.annotations.drop('age', axis = 1)

        self.weight = torch.tensor(np.exp(-self.annotations.iloc[:,1:].sum(0)/751))
        
        #max = np.max(self.annotations.iloc[:,1:].sum(0))
        #self.weight = torch.tensor(max / self.annotations.iloc[:,1:].sum(0)) / 101
        #self.weight[20] = 0
        self.img_name = os.listdir(img_path) # list all files in the images folder
        self.name = [i.split('_')[0] for i in self.img_name]
        self.names = [i.lstrip('0') for i in self.name] #The lstrip() method removes any leading characters (space is the default leading character to remove)

        self.df = pd.DataFrame({'id':self.names})
        self.df['filename'] = self.img_name
        #self.df = pd.DataFrame({'filename':self.img_name})
        #self.df['id'] = self.names
        self.df['id'] = self.df['id'].astype(int)
        self.df2 = pd.merge(self.df, self.annotations) #drop_duplicates(keep='first')
        # length
        self.data_len = len(self.img_name) 

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        #img_ID = self.df2.iloc[index,0]
        img_path = os.path.join(self.file_path, self.df2.iloc[index, 1])
        image = Image.open(img_path).convert('RGB')#io.imread(img_path)
        image = transform(image)#transforms.ToTensor(image)
        label = torch.tensor([self.df2.iloc[index,2:]], dtype=torch.float32)
        
        if self.transform is not None:
            image = self.transform (image)
            
        return (image, label)
        #return (image, img_ID)
        
    def __len__(self):

        return self.data_len


class testData(Dataset):
    def __init__(self, img_path, transform = None):
        self.file_path = img_path
        self.transform = transform 
        self.img_name = os.listdir(img_path) 
        self.data_len = len(self.img_name) 

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_path = os.path.join(self.file_path, self.img_name[index])#df.iloc[index, 1])
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        name = self.img_name[index]
        if self.transform is not None:
            image = self.transform (image)
            
        return (image, name)
        
    def __len__(self):

        return self.data_len

class testimage(Dataset):
    def __init__(self, img_path, transform = None):
        self.file_path = img_path
        self.transform = transform 
        self.data_len = 1#len(self.img_name) 

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = Image.open(self.file_path).convert('RGB')
        image = transform(image)
        if self.transform is not None:
            image = self.transform (image)
            
        return image
        
    def __len__(self):

        return self.data_len

'''
MyTestDataset = testData(
        #"classification_test.csv",
        "./dataset/test")

#print(len(MyTestDataset.weight))
print(MyTestDataset.img_name)

MyTrainDataset = ImagesData(
        "annotations_train.csv",
        "./dataset/train")
print(MyTrainDataset.annotations)#.iloc[:,1:].sum(0))
print(MyTrainDataset.weight)
#print(MyTrainDataset.__getitem__(512)[0])
'''
