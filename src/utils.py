"""Utils for PyTorch models and training"""
import numpy as np
import torch
import dataset
from torch import nn
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from networks import VGG


def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    new_lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Set lr to ', new_lr)
    return optimizer


def find_params():
    f = open('../data/CUBS/image_crop_labels.txt', 'w+') 
    cnn = VGG(200)
    num_scales = 2
    train_data = dataset.train_loader_cubs('../data/CUBS', 1, shuffle=False) 
    test_data = dataset.test_loader_cubs('../data/CUBS', 1, shuffle=False) 
    for data in [train_data, test_data]:
        for i, (img, label) in enumerate(data):
            img = torch.autograd.Variable(img) 
            img_id = data.dataset.image_ids[i]
            h, w = img.size(2), img.size(3) 
            hdiff = h - h//2
            wdiff = w - w//2
            split_centers = [(h//4, w//4), (h//4, w//2 + wdiff//2), 
                             (h//2 + hdiff//2, w//4),
                             (h//2 + hdiff//2, w//2, + wdiff//2)]
            centers = []
            for j in range(num_scales):
                best_split, best_center1 = get_best_split(img, cnn, split_centers)
                centers.append(best_center1)
                centers.append(h//4)
                _, best_center2 = get_best_split(best_split, cnn, split_centers)
                centers.extend(best_center2)
                centers.append(h//4)
            f.write("{} {} {} {} {} {} {}\n".format(img_id, *centers))
    f.close()


def get_best_split(img, model, split_centers):
    h, w = img.size(2), img.size(3) 
    splits = split_image(img)
    responses = [model(split)[1].data.norm() for split in splits]
    best = np.argmax(responses)
    return splits[best], split_centers[best]

        
def split_image(img):
    """
    :param img: (1, c, h, w)
    """
    h, w = img.size(2), img.size(3) 
    up = nn.Upsample(size=(h,w), mode='bilinear')
    tl = up(img[:, :, 0:h//2, 0:w//2])
    tr = up(img[:, :, 0:h//2, w//2:])
    bl = up(img[:, :, h//2:, 0:w//2])
    br = up(img[:, :, h//2:, w//2:])
    return tl, tr, bl, br
