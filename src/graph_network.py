import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import networks


dummy_input = Variable(torch.rand(1, 3, 224, 224))
model = networks.RACNN3(200, networks.VGG)
with SummaryWriter(comment='RACNN3') as w:
    w.add_graph(model, (dummy_input, ))
