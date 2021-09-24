import torch
from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn

import math

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])


        # Stage 1
        block1_1 = OrderedDict([
                        ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L1', [512, 1, 1, 1, 0])
                    ])
        blocks['block1_1'] = block1_1

        # self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv1_stage%d_L1' % i, [129, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 1, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        model0 = torchvision.models.vgg19(pretrained = True)
        
        # model0_1 = nn.Sequential(*list(model0.children())[:4])
        model0_1 = model0.features[:4]
        # print('01:')
        # print(model0_1)
        # model0_2 = nn.Sequential(*list(model0.children())[5:9])
        # model0_3 = nn.Sequential(*list(model0.children())[10:18])
        # model0_4 = nn.Sequential(*list(model0.children())[19:23])
        model0_2 = model0.features[5:9]
        # print(model0_2)
        model0_3 = model0.features[10:18]
        # print(model0_3)
        model0_4 = model0.features[19:23]
        # print(model0_4)
        self.model0 = model0
        self.model0_1 = model0_1
        self.model0_2 = model0_2
        self.model0_3 = model0_3
        self.model0_4 = model0_4
        self.myconv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.myconv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']


    def forward(self, x):
        x = x.to(torch.float32)
        outtemp = self.model0_1(x)
        # print(outtemp.shape)
        outtemp = self.model0_2(outtemp)
        # print(outtemp.shape)
        outtemp = self.model0_3(outtemp)
        # print(outtemp.shape)
        outtemp = self.model0_4(outtemp)
        # print(outtemp.shape)
        outtemp = self.myconv1(outtemp)
        out1 = self.myconv2(outtemp)

        out1_1 = self.model1_1(out1)
        out2 = torch.cat([out1_1, out1], 1)

        out2_1 = self.model2_1(out2)
        out3 = torch.cat([out2_1, out1], 1)

        out3_1 = self.model3_1(out3)
        out4 = torch.cat([out3_1, out1], 1)

        out4_1 = self.model4_1(out4)
        out5 = torch.cat([out4_1, out1], 1)

        out5_1 = self.model5_1(out5)
        out6 = torch.cat([out5_1, out1], 1)

        out6_1 = self.model6_1(out6)

        return out6_1
        # return outtemp

class My_L2loss(nn.Module):
    def __init__(self, sigma = 1.0):
        super(My_L2loss, self).__init__()
        self.sigma = sigma
    def forward(self, x, label): # label 是目标点的x，y坐标
        heatmap = torch.zeros(x.shape)
        mask = torch.zeros(x.shape)
        print(x.shape)
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                heatmap[0][0][i][j] = math.exp(- math.sqrt((i - label[0])*(i - label[0]) + (j - label[1])*(j - label[1])) / self.sigma/self.sigma)
                mask[0][0][i][j] = math.exp(- math.sqrt((i - label[0])*(i - label[0]) + (j - label[1])*(j - label[1])) / self.sigma/self.sigma)
        x = x.view(x.size(0), -1)
        
        print(heatmap)
        heatmap = heatmap.view(heatmap.size(0), -1)
        mask = mask.view(mask.size(0), -1)
        mask = mask.t()
        loss1 = (x - heatmap).pow(2)
        loss = loss1.mm(mask)
        return loss

