# --------------------------------------------------------
# Written by Shen Sang
# --------------------------------------------------------

import math
import torch
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class residualBlock(nn.Module):
    def __init__(self, nchannels=128):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=nchannels, out_channels=nchannels, kernel_size=3, dilation=2, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(nchannels)
        self.conv2 = nn.Conv2d(in_channels=nchannels, out_channels=nchannels, kernel_size=3, dilation=2, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(nchannels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), True)
        y = F.relu(self.bn2(self.conv2(y)), True)
        return y+x


class Encoder(nn.Module):
    def __init__(self, inc=4, bias=True, ngf=64):
        super(Encoder, self).__init__()
        self.en1 = nn.Sequential(nn.Conv2d(inc,     ngf,     (6, 6), 2, 2, bias=bias), nn.BatchNorm2d(ngf), nn.LeakyReLU(True))
        self.en2 = nn.Sequential(nn.Conv2d(ngf,     ngf * 2, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(True))
        self.en3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(True))
        self.en4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(True))
        self.en5 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(True))
        self.en6 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(True))
        self.en7 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(True))
        self.en8 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, (4, 4), 2, 1, bias=bias), nn.LeakyReLU(True))

    def forward(self, x):
        x1 = self.en1(x)
        x2 = self.en2(x1)
        x3 = self.en3(x2)
        x4 = self.en4(x3)
        x5 = self.en5(x4)
        x6 = self.en6(x5)
        x7 = self.en7(x6)
        x8 = self.en8(x7)

        return [x1, x2, x3, x4, x5, x6, x7, x8]


class DecoderRenderX3(nn.Module):
    def __init__(self, outc=3, img_size=256, ngf=64, light=3, bias=True, hard=True):
        super(DecoderRenderX3, self).__init__()
        self.light_mapping = nn.Sequential(nn.Linear(light, 32), nn.Tanh(), \
                                           nn.Linear(32, ngf * 2), nn.Tanh(), \
                                           nn.Linear(ngf * 2, ngf * 8), nn.Tanh())
        self.feat_fusion   = nn.Sequential(nn.Conv2d(ngf * 8 * 2, ngf * 8, (1, 1), bias=bias), nn.ReLU(True))

        self.ngf = ngf
        self.de1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 3, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 3, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de4 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 3, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de5 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 3, ngf * 4, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.de6 = nn.Sequential(nn.ConvTranspose2d(ngf * 4 * 3, ngf * 2, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 2), nn.ReLU(True))
        self.de7 = nn.Sequential(nn.ConvTranspose2d(ngf * 2 * 3, ngf, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf), nn.ReLU(True))
        self.de8 = nn.Sequential(nn.ConvTranspose2d(ngf * 3,     ngf, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf), nn.ReLU(True))
        
        self.res = nn.Sequential(*[residualBlock(ngf) for i in range(3)])
        if hard is True:
            self.out = nn.Sequential(nn.Conv2d(ngf, outc, (3, 3), 1, 1, bias=True), nn.Hardtanh())
        else:
            self.out = nn.Sequential(nn.Conv2d(ngf, outc, (3, 3), 1, 1, bias=True), nn.Tanh())

    def forward(self, feat_list, aux_feats, new_light):
        en1, en2, en3, en4, en5, en6, en7, en8 = feat_list
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = aux_feats

        bs = new_light.size(0)
        en8 = en8.view(bs, -1)

        light_vec = self.light_mapping(new_light)
        feat_concat = torch.cat((en8, light_vec), 1)
        feat_concat = feat_concat.view(bs, self.ngf * 8 * 2, 1, 1)
        feat_fusion = self.feat_fusion(feat_concat)

        de1 = self.de1(feat_fusion)
        de2 = self.de2(torch.cat((de1, en7, ax1), 1))
        de3 = self.de3(torch.cat((de2, en6, ax2), 1))
        de4 = self.de4(torch.cat((de3, en5, ax3), 1))
        de5 = self.de5(torch.cat((de4, en4, ax4), 1))
        de6 = self.de6(torch.cat((de5, en3, ax5), 1))
        de7 = self.de7(torch.cat((de6, en2, ax6), 1))
        de8 = self.de8(torch.cat((de7, en1, ax7), 1))
        
        res = self.res(de8)
        out = self.out(res)

        return out


class EnvmapPredictor(nn.Module):
    def __init__(self, numCoef=9):
        super(EnvmapPredictor, self).__init__()
        self.conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.bn = nn.InstanceNorm2d(1024)
        self.numCoef = numCoef

        self.regression = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(1024, self.numCoef * 3)
                )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, dim=2)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.regression(x))
        x = x.view(x.size(0), 3, self.numCoef)
        return x


class EnvmapPredictor2(nn.Module):
    def __init__(self, numCoef=9):
        super(EnvmapPredictor2, self).__init__()
        self.conv =nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, padding=0, stride=1)
        self.bn = nn.InstanceNorm2d(1024)
        self.numCoef = numCoef

        self.regression = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(1024, self.numCoef * 3)
                )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x) ) )
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, dim=2)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.regression(x))
        x = x.view( x.size(0), 3, self.numCoef)
        return x


class DecoderBRDFSingle(nn.Module):
    def __init__(self, outc=12, img_size=256, ngf=64, bias=True):
        super(DecoderBRDFSingle, self).__init__()

        self.ngf = ngf
        self.de1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de4 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.de5 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.de6 = nn.Sequential(nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf * 2), nn.ReLU(True))
        self.de7 = nn.Sequential(nn.ConvTranspose2d(ngf * 2 * 2, ngf, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf), nn.ReLU(True))
        self.de8 = nn.Sequential(nn.ConvTranspose2d(ngf * 2,     ngf, (4, 4), 2, 1, bias=bias), nn.BatchNorm2d(ngf), nn.ReLU(True))
        
        self.res = nn.Sequential(*[residualBlock(ngf) for i in range(1)])
        self.out = nn.Sequential(nn.Conv2d(ngf, outc, (3, 3), 1, 1, bias=True), nn.Tanh())

    def forward(self, feat_list):
        en1, en2, en3, en4, en5, en6, en7, en8 = feat_list

        de1 = self.de1(en8)
        de2 = self.de2(torch.cat((de1, en7), 1))
        de3 = self.de3(torch.cat((de2, en6), 1))
        de4 = self.de4(torch.cat((de3, en5), 1))
        de5 = self.de5(torch.cat((de4, en4), 1))
        de6 = self.de6(torch.cat((de5, en3), 1))
        de7 = self.de7(torch.cat((de6, en2), 1))
        de8 = self.de8(torch.cat((de7, en1), 1))
        
        res = self.res(de8)
        out = self.out(res)

        A = out[:, 0:3, :, :]
        N = out[:, 3:6, :, :]
        R = out[:, 6:9, :, :]
        D = out[:, 9: , :, :]

        albedo_pred = A
        norm = torch.sqrt(torch.sum(N * N, dim=1).unsqueeze(1)).expand_as(N)
        normal_pred = N / norm
        rough_pred = torch.mean(R, dim=1).unsqueeze(1)
        D = torch.mean(D, dim=1).unsqueeze(1)
        depth_pred = 1 / (0.4 * (D + 1) + 0.25)

        return [de1, de2, de3, de4, de5, de6, de7, de8], \
               [albedo_pred, normal_pred, rough_pred, depth_pred]


# -------------------------------Refine & Cascade---------------------------------------
class FusionBlock(nn.Module):
    def __init__(self, inc=7):
        super(FusionBlock, self).__init__()
        # relit: 3, render: 3, seg: 1, total = 7
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=64, kernel_size=6, stride=2, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.res1 = residualBlock(128)
        self.res2 = residualBlock(128)
        self.res3 = residualBlock(128)
        
        self.dres1 = residualBlock(128)
        self.dres2 = residualBlock(128)
        self.dres3 = residualBlock(128)
        self.dconv0 = nn.ConvTranspose2d(in_channels=128+128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.dbn0 = nn.BatchNorm2d(64)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64+64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.dbn1 = nn.BatchNorm2d(64)

        self.convFinal = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, imglist):
        img1 = imglist[-3]
        img2 = imglist[-2]
        seg = imglist[-1]

        x = torch.cat(imglist, 1)
        x1 = F.relu(self.bn1(self.conv1(x )))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.res1(x2)
        x3 = self.res2(x3)
        x3 = self.res3(x3)        
        
        x3 = self.dres1(x3)
        x3 = self.dres2(x3)
        x3 = self.dres3(x3)
        dx0 = F.relu(self.dbn0(self.dconv0(torch.cat((x3, x2), dim=1))), True)
        dx1 = F.relu(self.dbn1(self.dconv1(torch.cat((dx0, x1), dim=1))), True)
        out = F.softmax(self.convFinal(dx1), dim=1)

        w1 = out[:, 0, :, :].unsqueeze(1)
        w2 = out[:, 1, :, :].unsqueeze(1)
        
        final = img1 * w1 + img2 * w2
        return final


class RefineEncoder(nn.Module):
    def __init__(self):
        super(RefineEncoder, self).__init__()
        # render + relit + input + mask + lit = 13
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.res1 = residualBlock(128)
        self.res2 = residualBlock(128)
        self.res3 = residualBlock(128)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.res1(x2)
        x3 = self.res2(x3)
        x3 = self.res3(x3)
        return [x1, x2, x3]


class RefineDecoderRender(nn.Module):
    def __init__(self, litc=3):
        super(RefineDecoderRender, self).__init__()
        self.litc = litc
        self.dres1 = residualBlock(128+litc)
        self.dres2 = residualBlock(128+litc)
        self.dres3 = residualBlock(128+litc)
        self.dconv0 = nn.ConvTranspose2d(in_channels=128*2+litc, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(64)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64*3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(64)
        self.convFinal = nn.Conv2d(in_channels=64+64, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, feat_list, aux_feats, light):
        x1, x2, x3 = feat_list
        adx0, adx1 = aux_feats

        batch_size = light.size(0)
        light = light.unsqueeze(2).unsqueeze(3)
        light = light.expand((batch_size, self.litc, 64, 64))

        x3 = torch.cat([x3, light], dim=1)
        x3 = self.dres1(x3)
        x3 = self.dres2(x3)
        x3 = self.dres3(x3)
        dx0 = F.relu(self.dbn0(self.dconv0(torch.cat((x3, x2), dim=1))), True)
        dx1 = F.relu(self.dbn1(self.dconv1(torch.cat((dx0, x1, adx0), dim=1))), True)
        out = F.hardtanh(self.convFinal(torch.cat((dx1, adx1), dim=1)))
        return out


class RefineDecoderBRDF(nn.Module):
    def __init__(self):
        super(RefineDecoderBRDF, self).__init__()
        self.dres1 = residualBlock(128)
        self.dres2 = residualBlock(128)
        self.dres3 = residualBlock(128)
        self.dconv0 = nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(64)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(64)
        self.convFinal = nn.Conv2d(in_channels=64, out_channels=12, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, feat):
        x1, x2, x3 = feat
        x3 = self.dres1(x3)
        x3 = self.dres2(x3)
        x3 = self.dres3(x3)
        dx0 = F.relu(self.dbn0(self.dconv0(torch.cat((x3, x2), dim=1))), True)
        dx1 = F.relu(self.dbn1(self.dconv1(torch.cat((dx0, x1), dim=1))), True)
        out = F.tanh(self.convFinal(dx1))

        A = out[:, 0:3, :, :]
        N = out[:, 3:6, :, :]
        R = out[:, 6:9, :, :]
        D = out[:, 9: , :, :]

        albedo_pred = A
        norm = torch.sqrt(torch.sum(N * N, dim=1).unsqueeze(1)).expand_as(N)
        normal_pred = N / norm
        rough_pred = torch.mean(R, dim=1).unsqueeze(1)
        D = torch.mean(D, dim=1).unsqueeze(1)
        depth_pred = 1 / (0.4 * (D + 1) + 0.25)

        return [dx0, dx1], [albedo_pred, normal_pred, rough_pred, depth_pred]


class RefineDecoderEnv(nn.Module):
    def __init__(self):
        super(RefineDecoderEnv, self).__init__()
        self.numCoef = 9
        self.conv1 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.AvgPool2d(kernel_size = 4)
        self.projection = nn.Sequential(
                    nn.Linear(3 * self.numCoef, 512),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(512, 512)
                )
        self.regression = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(512, self.numCoef * 3)
                )


    def forward(self, x, pred):
        x = F.relu(self.bn1(self.conv1(x) ) )
        x = F.relu(self.bn2(self.conv2(x) ) )
        x = F.relu(self.bn3(self.conv3(x) ) )
        x = F.relu(self.bn4(self.conv4(x) ) )
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        pred = pred.view( pred.size(0), 3*self.numCoef)
        pred = self.projection(pred)
        x = torch.cat([pred, x], dim=1)
        x = torch.tanh(self.regression(x) )
        x = x.view(x.size(0), 3, self.numCoef)
        return x

# --------------------------------------------------------------

class encoderInitial(nn.Module):
    def __init__(self, intc=4):
        super(encoderInitial, self).__init__()
        # Input should be segmentation, image with environment map, image with point light + environment map
        self.conv1 = nn.Conv2d(in_channels=intc, out_channels=32, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)), True )
        x2 = F.relu(self.bn2(self.conv2(x1)), True )
        x3 = F.relu(self.bn3(self.conv3(x2)), True )
        x4 = F.relu(self.bn4(self.conv4(x3)), True )
        x5 = F.relu(self.bn5(self.conv5(x4)), True )
        x = F.relu(self.bn6(self.conv6(x5)), True )
        return x1, x2, x3, x4, x5, x


class decoderRender(nn.Module):
    def __init__(self, litc=3):
        super(decoderRender, self).__init__()
        self.light_mapping = nn.Sequential(nn.Linear(litc, 8), nn.Tanh(), \
                                           nn.Linear( 8,  32), nn.Tanh(), \
                                           nn.Linear(32, 128), nn.Tanh())
        # branch for normal prediction
        self.dconv0 = nn.ConvTranspose2d(in_channels=512+128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(256)
        self.dconv1 = nn.ConvTranspose2d(in_channels=256+256+256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(256)
        self.dconv2 = nn.ConvTranspose2d(in_channels=256+256+256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128+128+128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(in_channels=64+64+64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dconv5 = nn.ConvTranspose2d(in_channels=32+32+32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn5 = nn.BatchNorm2d(64)

        self.res = nn.Sequential(*[residualBlock(64) for i in range(3)])
        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 3, kernel_size = 5, stride=1, padding=2, bias=True)

    def forward(self, feat, feat_brdf, light):
        bs = light.size(0)

        x1, x2, x3, x4, x5, x = feat
        d1, d2, d3, d4, d5 = feat_brdf
        light_feat = self.light_mapping(light).view(bs, 128, 1, 1).expand(bs, 128, 4, 4)
        input = torch.cat([x, light_feat], dim=1)

        x_d1 = F.relu( self.dbn0(self.dconv0(input) ), True)
        x_d1_next = torch.cat( (x_d1, d1, x5), dim = 1)
        x_d2 = F.relu( self.dbn1(self.dconv1(x_d1_next) ), True)
        x_d2_next = torch.cat( (x_d2, d2, x4), dim = 1)
        x_d3 = F.relu( self.dbn2(self.dconv2(x_d2_next) ), True)
        x_d3_next = torch.cat( (x_d3, d3, x3), dim = 1)
        x_d4 = F.relu( self.dbn3(self.dconv3(x_d3_next) ), True)
        x_d4_next = torch.cat( (x_d4, d4, x2), dim = 1)
        x_d5 = F.relu( self.dbn4(self.dconv4(x_d4_next) ), True)
        x_d5_next = torch.cat( (x_d5, d5, x1), dim = 1)
        x_d6 = F.relu( self.dbn5(self.dconv5(x_d5_next) ), True)

        res = self.res(x_d6)
        out  = torch.tanh( self.convFinal(res) )
        return out


class decoderBRDF(nn.Module):
    def __init__(self):
        super(decoderBRDF, self).__init__()
        # branch for normal prediction
        self.dconv0 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(256)
        self.dconv1 = nn.ConvTranspose2d(in_channels=256+256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(256)
        self.dconv2 = nn.ConvTranspose2d(in_channels=256+256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128+128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(in_channels=64+64,  out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dconv5 = nn.ConvTranspose2d(in_channels=32+32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn5 = nn.BatchNorm2d(64)

        self.res = nn.Sequential(*[residualBlock(64) for i in range(1)])
        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 12, kernel_size = 5, stride=1, padding=2, bias=True)

    def forward(self, feat):
        x1, x2, x3, x4, x5, x = feat
        x_d1 = F.relu( self.dbn0(self.dconv0(x) ), True)
        x_d1_next = torch.cat( (x_d1, x5), dim = 1)
        x_d2 = F.relu( self.dbn1(self.dconv1(x_d1_next) ), True)
        x_d2_next = torch.cat( (x_d2, x4), dim = 1)
        x_d3 = F.relu( self.dbn2(self.dconv2(x_d2_next) ), True)
        x_d3_next = torch.cat( (x_d3, x3), dim = 1)
        x_d4 = F.relu( self.dbn3(self.dconv3(x_d3_next) ), True)
        x_d4_next = torch.cat( (x_d4, x2), dim = 1)
        x_d5 = F.relu( self.dbn4(self.dconv4(x_d4_next) ), True)
        x_d5_next = torch.cat( (x_d5, x1), dim = 1)
        x_d6 = F.relu( self.dbn5(self.dconv5(x_d5_next) ), True)
        out  = torch.tanh( self.convFinal(self.res(x_d6)) )

        A = out[:, 0:3, :, :]
        N = out[:, 3:6, :, :]
        R = out[:, 6:9, :, :]
        D = out[:, 9: , :, :]

        albedo_pred = A
        norm = torch.sqrt(torch.sum(N * N, dim=1).unsqueeze(1)).expand_as(N)
        normal_pred = N / norm
        rough_pred = torch.mean(R, dim=1).unsqueeze(1)
        D = torch.mean(D, dim=1).unsqueeze(1)
        depth_pred = 1 / (0.4 * (D + 1) + 0.25)

        return [x_d1, x_d2, x_d3, x_d4, x_d5], \
               [albedo_pred, normal_pred, rough_pred, depth_pred]


class envmapInitial(nn.Module):
    def __init__(self, numCoef=9):
        super(envmapInitial, self).__init__()
        self.conv =nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size=4, padding=0, stride=1)
        # self.bn = nn.BatchNorm2d(1024)
        self.bn = nn.InstanceNorm2d(1024)
        self.numCoef = numCoef

        self.regression = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(True),
                    nn.Dropout(0.25),
                    nn.Linear(1024, self.numCoef * 3)
                )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x) ) )
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, dim=2)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.regression(x))
        x = x.view( x.size(0), 3, self.numCoef)
        return x

# -----------------------------------------------------------------------------

# ------------------------ Ablation study ---------------------
class DecoderRenderX2(nn.Module):
    def __init__(self):
        super(DecoderRenderX2, self).__init__()
        self.light_mapping = nn.Sequential(nn.Linear( 3,   8), nn.Tanh(), \
                                           nn.Linear( 8,  32), nn.Tanh(), \
                                           nn.Linear(32, 128), nn.Tanh())
        # branch for normal prediction
        self.dconv0 = nn.ConvTranspose2d(in_channels=512+128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn0 = nn.BatchNorm2d(256)
        self.dconv1 = nn.ConvTranspose2d(in_channels=256+256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn1 = nn.BatchNorm2d(256)
        self.dconv2 = nn.ConvTranspose2d(in_channels=256+256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dconv3 = nn.ConvTranspose2d(in_channels=128+128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(in_channels=64+64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn4 = nn.BatchNorm2d(32)
        self.dconv5 = nn.ConvTranspose2d(in_channels=32+32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dbn5 = nn.BatchNorm2d(64)

        self.res = nn.Sequential(*[residualBlock(64) for i in range(3)])
        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 3, kernel_size = 5, stride=1, padding=2, bias=True)

    def forward(self, feat, light):
        bs = light.size(0)

        x1, x2, x3, x4, x5, x = feat
        light_feat = self.light_mapping(light).view(bs, 128, 1, 1).expand(bs, 128, 4, 4)
        input = torch.cat([x, light_feat], dim=1)

        x_d1 = F.relu( self.dbn0(self.dconv0(input) ), True)
        x_d1_next = torch.cat( (x_d1, x5), dim = 1)
        x_d2 = F.relu( self.dbn1(self.dconv1(x_d1_next) ), True)
        x_d2_next = torch.cat( (x_d2, x4), dim = 1)
        x_d3 = F.relu( self.dbn2(self.dconv2(x_d2_next) ), True)
        x_d3_next = torch.cat( (x_d3, x3), dim = 1)
        x_d4 = F.relu( self.dbn3(self.dconv3(x_d3_next) ), True)
        x_d4_next = torch.cat( (x_d4, x2), dim = 1)
        x_d5 = F.relu( self.dbn4(self.dconv4(x_d4_next) ), True)
        x_d5_next = torch.cat( (x_d5, x1), dim = 1)
        x_d6 = F.relu( self.dbn5(self.dconv5(x_d5_next) ), True)

        res = self.res(x_d6)
        out  = torch.tanh( self.convFinal(res) )
        return out