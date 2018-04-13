import torch
import numpy as np
import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    """
    VGG16 with Fine Grained Classification Head
    """
    def __init__(self, num_classes):
        super(VGGFeatures, self).__init__()
        base_model = models.vgg16(pretrained=True)
        base_features = base_model.features
        self.features = nn.Sequential(*base_features)
        self.n_features = 512 * 7 * 7
        self.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (num_batch, 3, h, w) np array batch of images to find class wise scores of
        :return: (num_batch, num_classes) np array of class wise scores per image
        """
        feats = self.features(x)
        feats = feats.view(x.size(0), -1)
        out = self.classifier(feats)

        return out, feats


class APN(nn.Module):
    """
    Attention Proposal Network
    """
    def __init__(self, n_features):
        super(APN, self).__init__()
        self.fc1 = nn.Linear(n_features, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, x):
        return self.fc1(self.fc2(x))


class CropUpscale(nn.Module):
    """
    Network that crops an image and upscales
    it to a standard size
    """
    def __init__(self, target_size):
        super(CropUpscale, self).__init__()
        self.up = nn.Upsample(size=target_size, mode='bilinear')

    def forward(self, x, crop_params):
        h, w = x.size(2), x.size(3)
        tx, ty, half_width = crop_params[0,0], crop_params[0, 1], crop_params[0, 2]
        txtl, txbr = tx - half_width, tx + half_width
        tytl, tybr = ty - half_width, ty + half_width
        xtl, xbr, ytl, ybr = self.get_noise_shifted_coords(h, w, txtl, txbr, tytl, tybr)

        Mx = torch.sigmoid(np.inf * xtl) - torch.sigmoid(np.inf * xbr)
        My = torch.sigmoid(np.inf * ytl) - torch.sigmoid(np.inf * ybr)
        M = torch.abs(torch.ger(Mx, My))
        masked_x = M * x
        crop_x = masked_x[:, :, th:bh, lw:rw]
        up_x = self.up(crop_x)
        return up_x

    def get_noise_shifted_coords(self, h, w, txtl, txbr, tytl, tybr):
        noise = 0.000001
        xs = torch.arange(0, h)
        ys = torch.arange(0, w)
        xtl = xs - txtl
        xbr = xs - txbr
        ytl = ys - tytl
        ybr = ys - tybr
        xtl[xtl == 0] += noise
        xbr[xbr == 0] -= noise
        ytl[ytl == 0] += noise
        ybr[ybr == 0] -= noise
        return xtl, xbr, ytl, ybr


class RACNN3(nn.Module):
    def __init__(self, num_classes, cnn):
        super(RACNN3, self).__init__()
        self.cnn1 = cnn(num_classes)
        self.apn1 = APN(self.cnn1.n_features)
        self.cnn2 = cnn(num_classes)
        self.apn2 = APN(self.cnn2.n_features)
        self.cnn3 = cnn(num_classes)
        self.cropup = CropUpscale((224, 224))

    def forward(self, x):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (1, 3, h, w) np array batch of images to find class wise scores of
        :return: (1, num_classes) np array of class wise scores per image
        """
        scores1, feats1 = self.cnn1(x)
        crop_params1 = self.apn1(feats1)
        crop_x = self.cropup(x, crop_params1)
        scores2, feats2 = self.cnn2(crop_x)
        crop_params2 = self.apn2(feats2)
        crop_x = self.cropup(x, crop_params2)
        scores3, _ = self.cnn3(x)
        return scores1, scores2, scores3


