import os
import torch
import torch.utils.data as data
import numpy as np
from torchvision.datasets.folder import default_loader
import re
import torchvision


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sorted(key=alphanum_key)


def check_name_patients(dir, seq_size):
    all_patient = [] 
    for subset in os.listdir(dir):
        for patient in os.listdir(dir +"/"+subset):
            num_image_patient = len(os.listdir(dir+"/"+subset+"/"+patient))
            if num_image_patient >= seq_size:
                #print(patient)
                all_patient.append(patient)
                            
    return all_patient

def check_number_slice(dir):
    all_files = []
    all_patient = []
    for index, split in enumerate(os.listdir(dir)): # split: test,train,val    
            for category in os.listdir(dir+"/"+split): #0:covid, 1:non-covid
                for subset in os.listdir(dir+"/"+ split + "/" + category): 
                    for patient in os.listdir(dir+"/"+ split + "/" + category+"/"+subset):
                            all_patient.append(patient)
                            num_files = len([f for f in os.listdir(os.path.join(dir, split + "/" + category+"/"+subset+"/"+ patient))])
                            all_files.append(num_files)
    return len(all_files), len(all_patient)

def _make_dataset(dir, seq_size):
    all_images = []
    all_labels = []
    count =0
    sub_middle_index = seq_size//2
    for index_label, category in enumerate(os.listdir(dir)): # 0:covid, 1:non-covid
            for subset in os.listdir(dir+"/"+category):
                for patient in os.listdir(dir+"/"+category+"/"+subset):
                    image_name_lst= []
                    img_lst = []
                    num_image_patient = len(os.listdir(dir+"/"+category+"/"+subset+"/"+patient))
                    middle_index=num_image_patient//2
                    if num_image_patient >= seq_size:
                        all_labels.append(index_label)
                        count_img_patient = 0
                        for image_name in sorted(os.listdir(dir+"/"+category+"/"+subset+"/"+patient), key=alphanum_key):
                            if(not image_name.startswith("._")):   
                                if(count_img_patient >= (middle_index-sub_middle_index) and count_img_patient <= (middle_index+sub_middle_index)):
                                    img_lst.append(dir+"/"+category+"/"+subset+"/"+patient+ "/"+image_name)
                                    image_name_lst.append(image_name)
                                count_img_patient+=1              
                    else:
                        continue                  
                    all_images.append(img_lst)
    return all_images, all_labels

class Cov19d_3DScan(data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, root, split = 'train', loader=default_loader, transform = None, seq_size = 1, input_dim= [300, 350]):
        self.root = root
        self.split = split
        self.loader = loader
        self.transform = transform
        self.seq_size = seq_size
        self.input_dim = input_dim
        self.imgs, self.targets = _make_dataset(os.path.join(self.root, self.split), self.seq_size)
        

    def __len__(self):
        'denotes the total number of samples'
        return len(self.imgs)

    def __getitem__(self, index):
        all_patient_images = self.imgs[index]
        all_patient_target = torch.tensor(self.targets[index], dtype=torch.float32)

        all_images = torch.Tensor(1, self.seq_size, self.input_dim[0], self.input_dim[1])
        
        start = 0
        for i in range(start, start + self.seq_size):
            image = torchvision.io.read_image(all_patient_images[i])
            
            if self.transform:
                image_transform = self.transform(image)

            all_images[0][i] = image_transform[0]

            start+=1
        return all_images, all_patient_target

def check_num_image_patient(dir):
    count_underscore_img = 0
    count_img = 0
    for image_name in sorted(os.listdir(dir), key=alphanum_key):
        if(image_name.startswith("._")):
            count_underscore_img+=1
        else:
            count_img+=1
    return count_img, count_underscore_img

def _make_dataset_eval(dir, seq_size):
    all_images = []
    patient_path = []
    count =0
    sub_middle_index = seq_size//2

    for subset in os.listdir(dir):
        for patient in os.listdir(dir+"/"+subset):
            image_name_lst= []
            img_lst = []
            num_underscore_image = 0
            num_image_patient, num_underscore_image = check_num_image_patient(dir+"/"+subset+"/"+patient)
            middle_index=num_image_patient//2

            if num_image_patient >= seq_size:
                patient_path.append(dir+"/"+subset+"/"+patient)
                count_img_patient = num_underscore_image
                for image_name in sorted(os.listdir(dir+"/"+subset+"/"+patient), key=alphanum_key):
                    if(not image_name.startswith("._")):   
                        if(count_img_patient >= (num_underscore_image+middle_index-sub_middle_index) and count_img_patient <= (num_underscore_image+middle_index+sub_middle_index)):
                            img_lst.append(dir+"/"+subset+"/"+patient+ "/"+image_name)
                            image_name_lst.append(image_name)
                        count_img_patient+=1              
            else:
                continue                
            all_images.append(img_lst)
    return all_images, patient_path

class Cov19d_3DScan_eval(data.Dataset):

    def __init__(self, root, split = 'train', loader=default_loader, transform = None, seq_size = 1, input_dim= [300, 350]):
        self.root = root
        self.split = split
        self.loader = loader
        self.transform = transform
        self.seq_size = seq_size
        self.input_dim = input_dim
        self.imgs, self.path = _make_dataset_eval(os.path.join(self.root, self.split), self.seq_size)
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.path[index].split('/')[-1]
        all_patient_images = self.imgs[index]

        all_images = torch.Tensor(1, self.seq_size, self.input_dim[0], self.input_dim[1])
        
        start = 0
        for i in range(start, start + self.seq_size):
            image = torchvision.io.read_image(all_patient_images[i])

            if self.transform:
                image_transform = self.transform(image)

            all_images[0][i] = image_transform[0]

            start+=1
        return all_images, img_path