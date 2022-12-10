import torch
import torch.nn as nn
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.ca = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1), #, bias=False
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.ca(self.avg_pool(x))
        max_out = self.ca(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        scale = self.sa(out)
        return x * scale


#resnet50
class res_net(nn.Module):
    def __init__(self, class_num = 32):
        super(res_net, self).__init__()
        #load pretrained model
        model_res = models.resnet50(pretrained=True)
        self.ca = ChannelAttention(in_planes=64)
        #self.ca1 = ChannelAttention(in_planes=2048)
        self.sa = SpatialAttention()
        self.model = model_res
        self.fc = nn.Linear(2048, class_num)
        
    def forward(self, x):
        
        #x = self.sa(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)        
        x = self.model.maxpool(x)

        x = self.ca(x)
        x = self.sa(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        
        #x = self.ca1(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x

class resnet(nn.Module):
    def __init__(self, class_num = 32):
        super(resnet, self).__init__()
        model = models.resnet50(pretrained=True)
        self.model = model
        self.fc = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        return x
#net = res_net()
#print(net)

