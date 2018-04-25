import torch
import numpy as np
import torch.nn as nn
from torchvision import models

IMSIZE = 224

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
    def __init__(self, num_classes, im_size):
        super(VGG, self).__init__()
        assert(im_size == 224 or im_size == 448)
        pool_size = 14 if im_size == 224 else 28

        base_model = models.vgg19(pretrained=True)
        base_features = list(base_model.features)
        self.features = [*base_features[:-2]]
        self.n_features = 512 * pool_size * pool_size
        self.flatten_features = View(-1, self.n_features)
        self.features = nn.Sequential(*self.features)
        base_classifier = list(base_model.classifier.children())
        fc6 = nn.Linear(512 * pool_size//2 * pool_size//2, 4096) if im_size == 448 else base_classifier[0]
        self.classifier = nn.Sequential(
                *base_features[-2:],
                View(-1, 512 * pool_size//2 * pool_size//2),
                fc6,
                *base_classifier[1:-1],
                nn.Linear(4096, num_classes)
        )
        for mod in self.classifier:
            if isinstance(mod, nn.ReLU):
                mod.inplace = False


    def forward(self, x):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (num_batch, 3, h, w) np array batch of images to find class wise scores of
        :return: (num_batch, num_classes) np array of class wise scores per image
        """
        feats = self.features(x)
        out = self.classifier(feats)

        return out, self.flatten_features(feats)


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
        self.regressor1 = nn.Tanh()
        self.fc2 = nn.Linear(1024, 3)
        self.regressor2 = nn.Sigmoid()

    def forward(self, x):
        params = self.regressor1(self.fc1(x))
        params = self.regressor2(self.fc2(params))
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
        assert(xtl.requires_grad and xbr.requires_grad and ytl.requires_grad and ybr.requires_grad)

        # Build crop mask and mask the image
        inf = torch.autograd.Variable(torch.Tensor([float('inf')]).cuda())
        Mx = torch.sigmoid(inf * xtl) - torch.sigmoid(inf * xbr)
        My = torch.sigmoid(inf * ytl) - torch.sigmoid(inf * ybr)
        M = torch.abs(torch.ger(Mx, My))
        masked_x = M * x
        assert(Mx.requires_grad and My.requires_grad and M.requires_grad and masked_x.requires_grad)

        # Crop out zeroed out values from mask
        tlx, brx = int(txtl.data[0]), int(txbr.data[0])
        tly, bry = int(tytl.data[0]), int(tybr.data[0])
        crop_x = masked_x[:, :, tlx:brx, tly:bry]
        up_x = self.up(crop_x)
        assert(up_x.requires_grad)

        return up_x

    def get_crop_corners(self, crop_params, h, w):
        tx, ty, half_width = crop_params[0,0], crop_params[0, 1], crop_params[0, 2]/2
        txtl = torch.clamp(tx - half_width, min=0).trunc()
        txbr = torch.clamp(tx + half_width, max=h-1).trunc()
        tytl = torch.clamp(ty - half_width, min=0).trunc()
        tybr = torch.clamp(ty + half_width, max=w-1).trunc()
        return txtl, txbr, tytl, tybr

    def get_noise_shifted_coords(self, h, w, txtl, txbr, tytl, tybr):
        noise = 0.000001
        xs = torch.autograd.Variable(torch.arange(0, h).cuda())
        ys = torch.autograd.Variable(torch.arange(0, w).cuda())
        xtl = xs - txtl.cuda()
        xbr = xs - txbr.cuda()
        ytl = ys - tytl.cuda()
        ybr = ys - tybr.cuda()
        '''
	The following code only isn't compatible with torch 0.30 and up at the moment
        Bug workaround below can be found at https://github.com/JannerM/intrinsics-network/issues/3
        xtl[xtl == 0] += noise
        xbr[xbr == 0] -= noise
        ytl[ytl == 0] += noise
        ybr[ybr == 0] -= noise
        '''
        xtl = xtl + ((xtl == 0) == 1).float() * noise
        xbr = xbr - ((xbr == 0) == 1).float() * noise
        ytl = ytl + ((ytl == 0) == 1).float() * noise
        ybr = ybr - ((ybr == 0) == 1).float() * noise

        return xtl, xbr, ytl, ybr

class RACNN2(nn.Module):
    def __init__(self, num_classes, cnn, init_apn=False):
        super(RACNN2, self).__init__()
        self.cnn1 = cnn(num_classes, IMSIZE)
        self.apn1 = APN(self.cnn1.n_features)
        self.cropup1 = CropUpscale((IMSIZE, IMSIZE))
        self.cnn2 = cnn(num_classes, IMSIZE)
        if init_apn:
            self.init_with_apn2()

    def forward(self, x, feats=False):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (1, 3, h, w) np array batch of images to find class wise scores of
        :return: (1, num_classes) np array of class wise scores per image
        """
        h, w = x.size(2), x.size(3)
        scores1, feats1 = self.cnn1(x)
        crop_params1 = self.apn1(feats1)

        crop_x1 = self.cropup1(x, h*crop_params1)
        scores2, feats2 = self.cnn2(crop_x1)

        if feats:
            return feats1, feats2

        return scores1, scores2

    def init_with_apn2(self):
        ckpt = torch.load('../checkpoints/CUBS/apn2.pt.pt')
        self.apn1.load_state_dict(ckpt['apn1_state_dict'])


class RACNN3(nn.Module):
    def __init__(self, num_classes, cnn, init_apn=False):
        super(RACNN3, self).__init__()
        self.cnn1 = cnn(num_classes, IMSIZE)
        self.apn1 = APN(self.cnn1.n_features)
        self.cropup1 = CropUpscale((IMSIZE, IMSIZE))
        self.cnn2 = cnn(num_classes, IMSIZE)
        self.apn2 = APN(self.cnn2.n_features)
        self.cropup2 = CropUpscale((IMSIZE, IMSIZE))
        self.cnn3 = cnn(num_classes, IMSIZE)
        if init_apn:
            self.init_with_apn2()

    def forward(self, x, feats=False):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (1, 3, h, w) np array batch of images to find class wise scores of
        :return: (1, num_classes) np array of class wise scores per image
        """
        h, w = x.size(2), x.size(3)
        scores1, feats1 = self.cnn1(x)
        crop_params1 = self.apn1(feats1)

        crop_x1 = self.cropup1(x, h*crop_params1)
        scores2, feats2 = self.cnn2(crop_x1)
        crop_params2 = self.apn2(feats2)

        crop_x2 = self.cropup2(crop_x1, h*crop_params2)
        scores3, feats3 = self.cnn3(crop_x2)

        if feats:
            return feats1, feats2, feats3

        return scores1, scores2, scores3

    def init_with_apn2(self):
        ckpt = torch.load('../checkpoints/CUBS/apn2.pt.pt')
        self.apn1.load_state_dict(ckpt['apn1_state_dict'])
        self.apn2.load_state_dict(ckpt['apn2_state_dict'])


class RACNN(nn.Module):
    def __init__(self, num_classes, cnn, init_racnn=False):
        super(RACNN, self).__init__()
        self.racnn = RACNN3(num_classes, cnn, False)
        self.avg_pool1 = nn.AvgPool2d(14, 14)
        self.avg_pool2 = nn.AvgPool2d(14, 14)
        self.avg_pool3 = nn.AvgPool2d(14, 14)
        self.fc1 = nn.Linear(1536, 200)
        self.fc2 = nn.Linear(1024, 200)
        if init_racnn:
            self.init_with_racnn()

    def forward(self, x):
        """
        Applies VGG16 forward pass for class wise scores
        :param input: (1, 3, h, w) np array batch of images to find class wise scores of
        :return: (1, num_classes) np array of class wise scores per image
        """
        feats1, feats2, feats3 = self.racnn(x, True)
        feats1 = 0.1 * self.avg_pool1(feats1).view(-1, 512)
        feats2 = 0.1 * self.avg_pool2(feats2).view(-1, 512)
        feats3 = 0.1 * self.avg_pool3(feats3).view(-1, 512)
        feats12 = torch.cat([feats1, feats2])
        feats123 = torch.cat([feats12, feats3])
        scores1 = self.fc1(feats12)
        scores2 = self.fc1(feats123)
        return scores1, scores2

    def init_with_racnn(self):
        ckpt = torch.load('../checkpoints/CUBS/racnn3.pt.pt')
        self.racnn.load_state_dict(ckpt['racnn3_state_dict'])


class APN2(nn.Module):
    def __init__(self, num_classes, cnn):
        super(APN2, self).__init__()
        self.cnn = cnn(num_classes, IMSIZE)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.apn1 = APN(self.cnn.n_features)
        self.apn2 = APN(self.cnn.n_features)
        self.cropup = CropUpscale((IMSIZE, IMSIZE))

    def forward(self, x, crop_params=None):
        h = x.size(2)
        _, feats = self.cnn(x)
        crop_params1 = self.apn1(feats)
        if crop_params is not None:
            cx, cy, hw = int(h*crop_params[0, 0]), int(h*crop_params[0, 1]), int(h*crop_params[0, 2])//2
            crop_x = x[:, :, cx-hw:cx+hw, cy-hw:cy+hw]
            crop_x = nn.Upsample(size=(h, h), mode='bilinear')(crop_x)
        else:
            crop_x = self.cropup(x, crop_params1*h)
        _, feats = self.cnn(crop_x)
        crop_params2 = self.apn2(feats)
        return torch.cat([crop_params1, crop_params2], 1)

