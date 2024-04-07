from __future__ import print_function

import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn
from PIL import Image
import torchvision.transforms.v2 as transforms  # I added the v2 to the import statement
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}
    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i  # here we are creating a dictionary with class names as keys and their index as values, kind of like enumerating the class names

    def __init__(self, split, size, data_dir='../data/VOCdevkit/VOC2007/'):
        super().__init__()
        self.split = split  # split is either 'train', 'val', 'trainval' or 'test', refer to the __init__ method to see how it is used
        self.data_dir = data_dir
        self.size = size  # size is the size of the image, it is a tuple of the form (H, W), refer to the __getitem__ method to see how it is used
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]  # this is a list of image names, one for each image in the split

        self.anno_list = self.preload_anno()

    @classmethod  # this is a class method, it is a method that is bound to the class and not the object of the class
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]  # returns the class name for a given index

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]  # returns the index for a given class name

    def __len__(self):
        return len(self.index_list)  # returns the number of images in the dataset

    def preload_anno(self):
        """
        :return: a list of labels. each element is in the form of [class, weight],
         where both class and weight are a numpy array in shape of [20],
        """
        label_list = []
        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)  # tree is an ElementTree object, it is a representation of the XML file in a tree structure

            #######################################################################
            # TODO: Insert your code here to preload labels
            # Hint: the folder Annotations contains .xml files with class labels
            # for objects in the image. The `tree` variable contains the .xml
            # information in an easy-to-access format (it might be useful to read
            # https://docs.python.org/3/library/xml.etree.elementtree.html)
            # Loop through the `tree` to find all objects in the image
            #######################################################################
            root = tree.getroot()  # root is the root element of the tree, it is the <annotation> tag in the XML file

            #  The class vector should be a 20-dimensional vector with class[i] = 1 if an object of class i is present in the image and 0 otherwise
            class_vec = torch.zeros(20)

            # The weight vector should be a 20-dimensional vector with weight[i] = 0 iff an object of class i has the `difficult` attribute set to 1 in the XML file and 1 otherwise
            # The difficult attribute specifies whether a class is ambiguous and by setting its weight to zero it does not contribute to the loss during training
            weight_vec = torch.ones(20)

            for obj in root.findall('object'):
                name = obj.find('name').text  # name is the class name of the object
                class_vec[self.get_class_index(name)] = 1  # setting the class_vec to 1 for the class index of the object
                difficult = int(obj.find('difficult').text)  # difficult is 1 if the object is difficult and 0 otherwise
                weight_vec[self.get_class_index(name)] = 1 - difficult  # setting the weight_vec to 1 - difficult for the class index of the object

            ######################################################################
            #                            END OF YOUR CODE                        #
            ######################################################################

            label_list.append((class_vec, weight_vec))

        return label_list

    def get_random_augmentations(self):
        ######################################################################
        # TODO: Return a list of random data augmentation transforms here
        # NOTE: make sure to not augment during test and replace random crops
        # with center crops. Hint: There are lots of possible data
        # augmentations. Some commonly used ones are random crops, flipping,
        # and rotation. You are encouraged to read the docs, which is found
        # at https://pytorch.org/vision/stable/transforms.html
        # Depending on the augmentation you use, your final image size will
        # change and you will have to write the correct value of `flat_dim`
        # in line 46 in simple_cnn.py
        ######################################################################

        trans = []  # list to store the transforms

        if self.split == 'train':
            # Random horizontal flipping
            trans.append(transforms.RandomHorizontalFlip(p=0.5))  # p is the probability of the image being flipped
            # Random rotation
            trans.append(transforms.RandomRotation(degrees=45))
            # Random cropping
            trans.append(transforms.RandomCrop(self.size))

        else:
            # Center cropping
            trans.append(transforms.CenterCrop(self.size))  # center cropping is used during testing

        return trans

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element
        image: FloatTensor in shape of (C, H, W) in scale [-1, 1].
        label: LongTensor in shape of (Nc, ) binary label
        weight: FloatTensor in shape of (Nc, ) difficult or not.
        """
        findex = self.index_list[index]  # findex is the name of the image file
        fpath = os.path.join(self.img_dir, findex + '.jpg')  # fpath is the path to the image file

        img = Image.open(fpath)  # img is a PIL image object, PIL is the Python Imaging Library

        trans = transforms.Compose([  # .Compose is a class that composes several transforms together
            transforms.Resize((self.size, self.size)),  # .Resize is a class that resizes the image to the given size
            *self.get_random_augmentations(),  # .get_random_augmentations() is a method that returns a list of random data augmentation transforms
            transforms.ToTensor(),  # .ToTensor is a class that converts a PIL image or numpy array to a tensor
            transforms.Normalize(mean=[0.485, 0.457, 0.407], std=[0.5, 0.5, 0.5]),  # .Normalize is a class that normalizes a tensor image with mean and standard deviation
            # mean and std are the mean and standard deviation of the ImageNet dataset
        ])

        img = trans(img)  # img is now a tensor image
        lab_vec, wgt_vec = self.anno_list[index]  # lab_vec is the class vector and wgt_vec is the weight vector for the image
        image = torch.FloatTensor(img)  # image is a tensor image
        label = torch.FloatTensor(lab_vec)  # label is a tensor label
        wgt = torch.FloatTensor(wgt_vec)  # wgt is a tensor weight

        return image, label, wgt  # returns the image, label, and weight for the image
