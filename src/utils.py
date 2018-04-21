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


def find_quad_params():
    f = open('../data/CUBS/image_crop_labels2.txt', 'w+') 
    cnn = VGG(200).cuda()
    num_scales = 2
    train_data = dataset.train_loader_cubs('../data/CUBS', 1, shuffle=False) 
    test_data = dataset.test_loader_cubs('../data/CUBS', 1, shuffle=False) 
    im_range = range(224//2)
    for data in [train_data, test_data]:
        for i, (img, label) in enumerate(data):
            img = torch.autograd.Variable(img.cuda()) 
            assert(img.size() == (1, 3, 224, 224))
            img_id = data.dataset.image_ids[i]
            h, w = img.size(2), img.size(3) 
            hdiff = h - h//2
            wdiff = w - w//2
            split_centers = [(h//2, w//2), (h//4, h//4),
                             (h//2 + hdiff//2, w//4), (h//4, w//2 + wdiff//2),
                             (h//2 + hdiff//2, w//2 + wdiff//2)]
            centers = []
            for j in range(num_scales):
                img, best_center = get_random_quad(img, cnn, im_range, im_range)
                centers.extend(best_center)
                centers.append(h//4)
            f.write("{} {} {} {} {} {} {}\n".format(img_id, *centers))
    f.close()


def get_best_quad(img, model, h_range, w_range):
    splits = split_image(img)
    responses = [model(split)[1].data.norm() for split in splits]
    best = np.argmax(responses)
    return splits[best], split_centers[best]


def get_random_quad(img, model, h_range, w_range):
    hs = np.random.choice(h_range, 5) 
    ws = np.random.choice(w_range, 5) 
    up = nn.Upsample(size=(224, 224), mode='bilinear')
    splits = [up(img[:, :, h:h+224//2, w:w+224//2]) for h, w in zip(hs, ws)]
    responses = [model(split)[1].data.norm() for split in splits]
    best = np.argmax(responses)
    return splits[best], (hs[best], ws[best])


def split_image(img):
    """
    :param img: (1, c, h, w)
    """
    h, w = img.size(2), img.size(3) 
    up = nn.Upsample(size=(h,w), mode='bilinear')
    c = up(img[:, :, h//4:3*h//4, w//4:3*w//4])
    tl = up(img[:, :, 0:h//2, 0:w//2])
    tr = up(img[:, :, 0:h//2, w//2:])
    bl = up(img[:, :, h//2:, 0:w//2])
    br = up(img[:, :, h//2:, w//2:])
    return c, tl, tr, bl, br


def find_best_params():
    f = open('../data/CUBS/image_crop_labels2.txt', 'w+') 
    cnn = VGG(200).cuda()
    num_scales = 2
    train_data = dataset.train_loader_cubs('../data/CUBS', 1, shuffle=False) 
    test_data = dataset.test_loader_cubs('../data/CUBS', 1, shuffle=False) 
    h = 224
    w = 224
    up = nn.Upsample(size=(h,w), mode='bilinear')
    for data in [train_data, test_data]:
        for i, (img, label) in enumerate(data):
            img = torch.autograd.Variable(img.cuda()) 
            img_id = data.dataset.image_ids[i]
            for j in range(num_scales):
                img, best_center = get_best_split(img, cnn, up)
                centers.extend(best_center)
                centers.append(h//4)
            f.write("{} {} {} {} {} {} {}\n".format(img_id, *centers))
    f.close()


def get_best_split(img, model, up):
    h, w = img.size(2), img.size(3) 
    max_response = 0
    max_center = None
    max_crop = None
    for i in range(h - h//2):
        for j in range(w - w//2):
            endi, endj = i + h//2, j + w//2
            crop = up(img[:, :, i:endi, j:endj])
            if model(crop)[1].data.norm() > max_response:
                max_center = (i + h//4, j + h//4)
                max_crop = crop
    return max_crop, max_center
        
if __name__ == '__main__':
    find_quad_params()
