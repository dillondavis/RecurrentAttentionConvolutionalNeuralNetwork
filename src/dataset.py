import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os.path as osp
import numpy as np
import torch
import glob
import os
from PIL import Image

IMSIZE = 224

class CUBS2011(data.Dataset):
    def __init__(self, root, split='train', transform=False, coords=False):
        self.root = root
        self.split = split
        self.coords = coords
        self._transform = transform
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.bw_image_ids = ['1401', '3617', '3780', '5393', '448', '3619', '5029', '6321']
        self.image_ids = self.get_image_ids()
        self.id_to_file = self.get_id_to_file()
        self.im_transform = transforms.Compose([
            transforms.Resize((IMSIZE, IMSIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean, std=self.std
            )
        ])
        if coords:
            self.id_to_coords = self.get_id_to_coords()
        else:
            self.id_to_label = self.get_id_to_label()

    def get_image_ids(self):
        image_ids = []
        split_file = os.path.join(self.root, 'train_test_split.txt')
        desired_split = '1' if self.split == 'train' else '0'
        with open(split_file) as f:
            for line in f:
                split = line.split()
                if split[1] == desired_split and split[0] not in self.bw_image_ids:
                    image_ids.append(split[0])
        return image_ids

    def get_id_to_file(self):
        id_to_file = {}
        img_file = os.path.join(self.root, 'images.txt')
        with open(img_file) as f:
            for line in f:
                split = line.split()
                id_to_file[split[0]] = split[1]
        return id_to_file

    def get_id_to_label(self):
        id_to_label = {}
        lbl_file = os.path.join(self.root, 'image_class_labels.txt')
        with open(lbl_file) as f:
            for line in f:
                split = line.split()
                id_to_label[split[0]] = int(split[1])
        return id_to_label

    def get_id_to_coords(self):
        id_to_coords = {}
        lbl_file = os.path.join(self.root, 'image_crop_labels_random2.txt')
        with open(lbl_file) as f:
            for line in f:
                split = line.split()
                id_to_coords[split[0]] = np.array([int(i) for i in split[1:]])
        return id_to_coords

    def find_invalid_bw_images(self):
        for i in self.image_ids:
            image_file = self.id_to_file[i]
            image_path = os.path.join(self.root, 'images/'+image_file)
            img = Image.open(image_path)
            if len(np.array(img).shape) != 3:
                print(i)
                print(img.mode)
                print(image_file)
                print(np.array(img).shape)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_file = self.id_to_file[image_id]
        image_path = os.path.join(self.root, 'images/'+image_file)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RBG')
        if self.coords:
            lbl = self.id_to_coords[image_id]
        else:
            lbl = self.id_to_label[image_id] - 1

        if self._transform:
            return self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        new_img = self.im_transform(img)
        if self.coords:
            lbl = lbl.astype(float) / IMSIZE

        return new_img, lbl


def train_loader_cubs(path, batch_size, num_workers=4, pin_memory=False, normalize=None, transform=True, shuffle=True, coords=False):
    return data.DataLoader(CUBS2011(path, transform=transform, coords=coords), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def test_loader_cubs(path, batch_size, num_workers=4, pin_memory=False, normalize=None, transform=True, shuffle=True, coords=False):
    return data.DataLoader(CUBS2011(path, split='test', transform=transform, coords=coords), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
