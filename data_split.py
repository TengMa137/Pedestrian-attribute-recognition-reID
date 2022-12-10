import os
from shutil import copyfile
from shutil import rmtree
import numpy as np
#import pandas as pd

def build_folder(dataDir): #put images from different people in different folders
    save_path = dataDir    #save_path = '/home/teng/DLproject/dataset'
    if not os.path.isdir(save_path): os.mkdir(save_path)

    train_save_path = save_path + '/train_all'
    if not os.path.isdir(train_save_path): os.mkdir(train_save_path)

    for root, dirs, files in os.walk('train', topdown=True):
        for name in files:
            ID  = name.split('_')
            src_path = 'train' + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
def get_valid(dataDir):  #divide train/val for pedestrain attributes recognition
    
    sourceDir = dataDir + '/train_all'  #dataDir = './dataset'
    trainDir = dataDir + '/train'      
    valDir = dataDir + '/val'          

    rmtree(trainDir) #delete folder if you already had one
    rmtree(valDir)
    os.mkdir(trainDir)
    os.mkdir(valDir)
    folders = os.listdir(sourceDir)
    folder_list = list(range(751))
    val_list = np.random.choice(folder_list, 151, replace=False)
    train_list = [e for e in folder_list if e not in val_list]
    for i in val_list:
        filelist = os.listdir(os.path.join(sourceDir, folders[i])) 
        for file in filelist:
            copyfile(sourceDir + '/' + folders[i] + '/' + file, valDir + '/' + file)
    for j in train_list:
        filelist = os.listdir(os.path.join(sourceDir, folders[j]))
        for file in filelist:
            copyfile(sourceDir + '/' + folders[j] + '/' + file, trainDir + '/' + file)

#---------------------------------------
def get_valid_reid(dataDir):  ##divide train/val for reid 

    train_save_path = dataDir + '/train_reid'  #'./dataset/train_reid'
    val_save_path = dataDir + '/val_reid'  
    if not os.path.isdir(train_save_path): os.mkdir(train_save_path)
    if not os.path.isdir(val_save_path): os.mkdir(val_save_path)

    for root, dirs, files in os.walk(dataDir + '/val', topdown=True):
        for name in files:

            ID  = name.split('_')
            src_path = dataDir + '/val/' + name
            dst_path = train_save_path + '/' + ID[0]
            
            if not os.path.isdir(dst_path): #first image is used as val image
                os.mkdir(dst_path)          
                copyfile(src_path, val_save_path + '/' + name)
            else:
                copyfile(src_path, train_save_path + '/' + name)
        
    for root, dirs, files in os.walk(train_save_path): #delete empty folders
        if not os.listdir(root): os.rmdir(root)
    
#get_valid_reid('./dataset')

