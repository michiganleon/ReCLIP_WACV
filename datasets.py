import os
import csv
import socket
import torchvision
import pandas
import json
import scipy.io
import torch
from os import path
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from torchvision.datasets import Food101, CIFAR10, CIFAR100, StanfordCars
from torchvision.datasets import FGVCAircraft, VOCDetection, DTD, OxfordIIITPet, Caltech101
from torchvision.datasets import EuroSAT, GTSRB, Kitti, Country211, PCAM, Kinetics, RenderedSST2
from torchvision.datasets import UCF101, FER2013, ImageNet, Flowers102, MNIST, STL10
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from urllib.error import HTTPError


# SETUP DATASET FOLDER HERE!!!!
if(socket.gethostname()[-7:] == 'usc.edu'):
    data_dir = "/project/nevatia_174/xuefeng_files/clip_datasets"
else:
    data_dir = "path/to/your/data/folder"


# default transformation for debugging the code
ImageNetTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

pytorch_implemented_sets = [
    'food101',
    'cifar10',
    'cifar100',
    'standford_cars',
    'fgvc',
    'dtd',
    'oxford_pets',
    'flowers102',
    'mnist',
    'stl10',
    'eurosat',
    'gtsrb',
    'country211',
    'pcam',
    'renderedsst2',
    'caltech101',
    # 'ucf101',
    # 'kitti',
    # 'k700',
    # 'voc2007',
    # 'cifar10_train',
    # 'cifar100_train',
]

self_implemented_sets = [
    'fer2013',
    'imagenet',
    'birdsnap',
    'resisc45',
    'aid',
    'sun397',
    'office_d',
    'office_a',
    'office_w',
    'office_ar',
    'office_cl',
    'office_pr',
    'office_rw',
    # 'aid_train',
    # 'sun397_train',
]


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

class pytorch_dataset_wrapper(Dataset):
    def __init__(self, name, preprocess=ImageNetTransform):
        assert(name in pytorch_implemented_sets)
        if(name == 'food101'):
            self.dataset = Food101(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'cifar10'):
            self.dataset = CIFAR10(root=data_dir, train=False, transform=preprocess, download=True)
        elif(name == 'cifar100'):
            self.dataset = CIFAR100(root=data_dir, train=False, transform=preprocess, download=True)
        elif(name == 'cifar10_train'):
            self.dataset = CIFAR10(root=data_dir, train=True, transform=preprocess, download=True)
        elif(name == 'cifar100_train'):
            self.dataset = CIFAR100(root=data_dir, train=True, transform=preprocess, download=True)
        elif(name == 'standford_cars'):
            self.dataset = StanfordCars(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'fgvc'):
            self.dataset = FGVCAircraft(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'dtd'):
            self.dataset = DTD(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'oxford_pets'):
            self.dataset = OxfordIIITPet(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'caltech101'):
            self.dataset = Caltech101(root=data_dir, transform=preprocess, download=True) # split
        elif(name == 'flowers102'):
            self.dataset = Flowers102(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'mnist'):
            self.dataset = MNIST(root=data_dir, train=False, transform=preprocess, download=True)
        elif(name == 'stl10'):
            self.dataset = STL10(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'eurosat'):
            self.dataset = EuroSAT(root=data_dir, transform=preprocess, download=True) # split
        elif(name == 'gtsrb'):
            self.dataset = GTSRB(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'country211'):
            self.dataset = Country211(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'pcam'):
            self.dataset = PCAM(root=data_dir, split='val', transform=preprocess, download=True)
        elif(name == 'renderedsst2'):
            self.dataset = RenderedSST2(root=data_dir, split='test', transform=preprocess, download=True)
        elif(name == 'imagenet'):
            self.dataset = ImageNet(root=data_dir, split='val', transform=preprocess)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, idx, label


# self-implemented datasets
class AID(Dataset):
    def __init__(self, transform=ImageNetTransform, root_dir=data_dir, train=False):
        self.transform = transform
        if(train):
            self.root_dir = root_dir + '/AID/AID_full/train/'
            self.label_file = pandas.read_feather(root_dir + '/AID/labels_full/labels_train.feather')
        else:
            self.root_dir = root_dir + '/AID/AID_full/test/'
            self.label_file = pandas.read_feather(root_dir + '/AID/labels_full/labels_test.feather')
        self.classes = sorted(list(set(self.label_file['class'])))
        self.files = self.label_file['id']
        self.labels = self.label_file['class']
        self.class2label = {self.classes[i]:i for i in range(len(self.classes))}

    def __len__(self):
        return(len(self.label_file['class']))

    def __getitem__(self, idx):
        img = Image.open(self.root_dir + self.files[idx]).convert('RGB')
        label = self.class2label[self.labels[idx]]
        return self.transform(img), idx, label

class resisc45(Dataset):
    def __init__(self, transform=ImageNetTransform, root_dir=data_dir):
        self.root_dir = root_dir + '/resisc45/RESISC45_full/test/'
        self.transform = transform
        self.label_file = pandas.read_feather(root_dir + '/resisc45/labels_full/labels_test.feather')
        self.classes = sorted(list(set(self.label_file['class'])))
        self.files = self.label_file['id']
        self.labels = self.label_file['class']
        self.class2label = {self.classes[i]:i for i in range(len(self.classes))}
        # print(self.classes)

    def __len__(self):
        return(len(self.label_file['class']))

    def __getitem__(self, idx):
        img = Image.open(self.root_dir + self.files[idx]).convert('RGB')
        label = self.class2label[self.labels[idx]]
        return self.transform(img), idx, label

class imagenet(Dataset):
    def __init__(self, transform=ImageNetTransform, root_dir=data_dir):
        self.root_dir = root_dir + '/imagenet/'
        self.transform = transform
        
        # imagenet class names
        name_table = json.load(open(self.root_dir + 'imagenet_categories.json','rb'))
        self.classes = [name_table[str(i)][1] for i in range(1000)]
        self.id2label = {name_table[str(i)][0]:i for i in range(1000)}

        # validation set labels
        mat = scipy.io.loadmat(os.path.join(self.root_dir, 'ILSVRC2012_devkit_t12/data/meta.mat'))['synsets']
        label_file = open(os.path.join(self.root_dir, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
        self.labels = [self.id2label[mat[int(x)-1][0][1][0]] for x in label_file.readlines()]
        
        self.files = os.listdir(os.path.join(self.root_dir, 'validation'))
        self.files.sort()
        
    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        fn = os.path.join(self.root_dir, 'validation', self.files[idx])
        img = Image.open(fn).convert('RGB')
        label = self.labels[idx]
        return self.transform(img), idx, label

class SUN397(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root_dir=data_dir, train=False):
        self.root_dir = root_dir + '/SUN397'
        self.transform = transform
        
        with open(self.root_dir+'/ClassName.txt','r') as file:
            classnames = file.read().splitlines()
            self.class2id = {classnames[i]:i for i in range(len(classnames))}
        
        if(train):
            with open(self.root_dir+'/Training.txt','r') as file:
                test_files = file.read().splitlines()
                self.files = test_files
        else:
            with open(self.root_dir+'/Testing_01.txt','r') as file:
                test_files = file.read().splitlines()
                self.files = test_files
                
        self.labels = []
        for i in range(len(self.files)):
            if(train):
                curr_class_name = '/'+'/'.join(self.files[i].split('/')[6:-1])
            else:
                curr_class_name = '/'.join(self.files[i].split('/')[:-1])
            curr_label = self.class2id[curr_class_name]
            self.labels.append(curr_label)
        
    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        img = Image.open(self.root_dir+self.files[idx]).convert('RGB')
        label = self.labels[idx]
        return self.transform(img), idx, label

class fer2013(Dataset):
    def __init__(self, transform=ImageNetTransform, root_dir=data_dir):
        self.root_dir = root_dir + '/fer2013'
        self.transform = transform
        
        self.images = []
        self.labels = []
        with open(self.root_dir + '/public_test.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                label = int(row[0])
                data = row[1].split(' ')
                data = np.array([int(x) for x in data])
                data = np.reshape(data, (48,48))
                data = Image.fromarray(np.uint8(data)).convert('RGB')
                self.images.append(data)
                self.labels.append(label)
        
    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        return self.transform(img), idx, label

class birdsnap(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root_dir=data_dir):
        self.root_dir = root_dir + '/birdsnap'
        self.transform = transform
        
        # available test images
        with open(root_dir + '/birdsnap/test_images.txt','r') as file:
            test_images = file.read().splitlines()[1:]
        self.available_files = []
        for filenames in test_images:
            full_file_name = root_dir + '/birdsnap/download/images/' + filenames
            if(path.exists(full_file_name)):
                self.available_files.append(full_file_name)
    
        # get folders
        with open(root_dir + '/birdsnap/species.txt','r') as file:
            species = file.read().splitlines()[1:]
    
        names = []
        folders = []
        for line in species:
            line_split = line.split('\t')
            names.append(line_split[1])
            folders.append(line_split[3])
    
        new_order = np.argsort(names)
        self.names = [names[i] for i in new_order]
        self.folders = [folders[i] for i in new_order]
    
        self.folder2id = {self.folders[i]:i for i in range(len(folders))}
     
        # get labels 
        self.labels = []
        for file in self.available_files:
            folder_name = file.split('/')[-2]
            curr_id = self.folder2id[folder_name.lower()]
            self.labels.append(curr_id)

    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        img = Image.open(self.available_files[idx])
        label = self.labels[idx]
        return self.transform(img), idx, label


class caltech101(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root_dir=data_dir):
        self.root_dir = root_dir + '/caltech101/101_ObjectCategories'
        self.transform = transform
        
        self.categories = os.listdir(self.root_dir)
        self.categories.sort()
        
        self.images = []
        self.labels = []
        
        with open(root_dir+'/caltech101/caltech101_test','r') as file:
            self.test_dict = json.load(file)
        
        for i in range(len(self.categories)):
            files = self.test_dict[self.categories[i].lower()]
            for filename in files:
                full_file_name = os.path.join(self.root_dir, self.categories[i], filename)
                if os.path.exists(full_file_name):
                    self.images.append(full_file_name)
                    self.labels.append(i)
                else:
                    print('missing file!')
        

    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        return self.transform(img), idx, label
    
class office(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root_dir=data_dir, mode='dslr'):
        assert(mode in ['dslr','amazon','webcam'])
        self.root_dir = root_dir + '/office31/'+mode+'/images/'
        self.transform = transform
        
        self.categories = os.listdir(self.root_dir)
        self.categories.sort()
        
        self.images = []
        self.labels = []
        
        for i in range(len(self.categories)):
            files = os.listdir(path.join(self.root_dir,self.categories[i]))
            for file in files:
                self.images.append(path.join(self.root_dir,self.categories[i],file))
                self.labels.append(i)
    
    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        return self.transform(img), idx, label

class officehome(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root_dir=data_dir, mode='dslr'):
        assert(mode in ['Art','Clipart','Product', 'Real_World'])
        self.root_dir = root_dir + '/officehome/'+mode
        self.transform = transform
        
        self.categories = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
        
        self.images = []
        self.labels = []
        
        for i in range(len(self.categories)):
            files = os.listdir(path.join(self.root_dir,self.categories[i]))
            for file in files:
                self.images.append(path.join(self.root_dir,self.categories[i],file))
                self.labels.append(i)
    
    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        return self.transform(img), idx, label


def get_dataset(dataset_name, preprocess):
    print(dataset_name)
    # pytorch implemented dataset
    if(dataset_name in pytorch_implemented_sets):
        return pytorch_dataset_wrapper(dataset_name, preprocess)
    # other
    assert(dataset_name in self_implemented_sets)
    if(dataset_name == 'aid'):
        return AID(transform=preprocess)
    elif(dataset_name == 'aid_train'):
        return AID(transform=preprocess, train=True)
    elif(dataset_name == 'imagenet'):
        return imagenet(transform=preprocess)
    elif(dataset_name == 'fer2013'):
        return fer2013(transform=preprocess)
    elif(dataset_name == 'birdsnap'):
        return birdsnap(transform=preprocess)
    elif(dataset_name == 'resisc45'):
        return resisc45(transform=preprocess)
    elif(dataset_name == 'sun397'):
        return SUN397(transform=preprocess)
    elif(dataset_name == 'sun397_train'):
        return SUN397(transform=preprocess, train=True) 
    elif(dataset_name == 'caltech101'):
        return caltech101(transform=preprocess)
    elif(dataset_name == 'office_d'):
        return office(transform=preprocess, mode='dslr')
    elif(dataset_name == 'office_a'):
        return office(transform=preprocess, mode='amazon')
    elif(dataset_name == 'office_w'):
        return office(transform=preprocess, mode='webcam')
    elif(dataset_name == 'office_ar'):
        return officehome(transform=preprocess, mode='Art')
    elif(dataset_name == 'office_cl'):
        return officehome(transform=preprocess, mode='Clipart')
    elif(dataset_name == 'office_pr'):
        return officehome(transform=preprocess, mode='Product')
    elif(dataset_name == 'office_rw'):
        return officehome(transform=preprocess, mode='Real_World')
