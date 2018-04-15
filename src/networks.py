import torch
import numpy as np
import torch.nn as nn
from torchvision import models


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class VGG(nn.Module):
    """
    VGG16 with Fine Grained Classification Head
    """
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        base_model = models.vgg16(pretrained=True)
        base_features = base_model.features
        self.features = [*base_features, View(-1, 25088)]
        self.features.extend(list(base_model.classifier.children())[:-1])
        self.n_features = 4096
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(self.n_features, num_classes)


    def forward(self, x):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (num_batch, 3, h, w) np array batch of images to find class wise scores of
        :return: (num_batch, num_classes) np array of class wise scores per image
        """
        feats = self.features(x)
        out = self.classifier(feats)

        return out, feats


class ResNet(nn.Module):
    """
    ResNet50 with Fine Grained Classification Head
    """
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        base_model = models.resnet50(pretrained=True)
        base_features = [x for x in base_model.children()][:-1]
        self.n_features = 2048 * 7 * 7
        self.features = nn.Sequential(*base_features)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_features, num_classes),
        )

    def forward(self, x):
        """
        Applies ResNet forward pass for class wise scores
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
        self.regressor = nn.Tanh()

    def forward(self, x, im_size):
        params = self.fc2(self.fc1(x))
        params = (self.regressor(params) + 1) / 2
        params *= im_size 
        return params


class CropUpscale(nn.Module):
    """
    Network that crops an image and upscales
    it to a standard size
    """
    def __init__(self, target_size):
        super(CropUpscale, self).__init__()
        self.up = nn.Upsample(size=target_size, mode='bilinear')

    def forward(self, x, crop_params):
        # Get crop corner coords and shifted x and y range of image
        h, w = x.size(2), x.size(3)
        txtl, txbr, tytl, tybr = self.get_crop_corners(crop_params, h, w)
        xtl, xbr, ytl, ybr = self.get_noise_shifted_coords(h, w, txtl, txbr, tytl, tybr)

        # Build crop mask and mask the image
        Mx = torch.sigmoid(np.inf * xtl) - torch.sigmoid(np.inf * xbr)
        My = torch.sigmoid(np.inf * ytl) - torch.sigmoid(np.inf * ybr)
        M = torch.abs(torch.ger(Mx, My))
        masked_x = M * x

        # Crop out zeroed out values from mask
        tlx, brx = int(txtl.data[0]), int(txbr.data[0])
        tly, bry = int(tytl.data[0]), int(tybr.data[0])
        crop_x = masked_x[:, :, tlx:brx, tly:bry]
        up_x = self.up(crop_x)
        return up_x

    def get_crop_corners(self, crop_params, h, w):
        tx, ty, half_width = crop_params[0,0], crop_params[0, 1], crop_params[0, 2]/2
        txtl = torch.clamp(tx - half_width, min=0)
        txbr = torch.clamp(tx + half_width, max=h-1)
        tytl = torch.clamp(ty - half_width, min=0)
        tybr = torch.clamp(ty + half_width, max=w-1)
        return txtl, txbr, tytl, tybr

    def get_noise_shifted_coords(self, h, w, txtl, txbr, tytl, tybr):
        noise = 0.000001
        xs = torch.autograd.Variable(torch.arange(0, h).cuda())
        ys = torch.autograd.Variable(torch.arange(0, w).cuda())
        xtl = xs - txtl
        xbr = xs - txbr
        ytl = ys - tytl
        ybr = ys - tybr
        ''' 
	The following code only isn't compatible with torch 0.30 and up at the moment
        Bug workaround can be found at https://github.com/JannerM/intrinsics-network/issues/3
        '''
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
        h, w = x.size(2), x.size(3)
        scores1, feats1 = self.cnn1(x)
        crop_params1 = self.apn1(feats1, h)
        crop_x = self.cropup(x, crop_params1)
        scores2, feats2 = self.cnn2(crop_x)
        crop_params2 = self.apn2(feats2, h)
        crop_x = self.cropup(x, crop_params2)
        scores3, _ = self.cnn3(x)
        return scores1, scores2, scores3

    def flip_apns(self):
        for apn in [self.apn1, self.apn2]:
            for param in apn.parameters():
                param.requires_grad = not param.requires_grad

    def flip_cnns(self):
        for cnn in [self.cnn1, self.cnn2, self.cnn3]:
            for param in cnn.parameters():
                param.requires_grad = not param.requires_grad

