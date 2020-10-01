import math
import torch
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class RenderLayerPointLightTorch():
    def __init__(self, imSize=256, fov=60, F0=0.05, cameraPos=[0, 0, 0], 
            lightPower=5.95, isCuda=True):
        self.imSize = imSize
        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = torch.from_numpy(np.array(cameraPos, \
                                dtype=np.float32).reshape([1, 3, 1, 1]))
        self.lightPower = lightPower
        self.yRange = self.xRange = 1 * np.tan(self.fov/2)
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imSize),
                np.linspace(-self.yRange, self.yRange, imSize) )
        y = np.flip(y, axis=0)
        z = -np.ones((imSize, imSize), dtype=np.float32)

        pCoord = torch.from_numpy(np.stack([x, y, z]).astype(np.float32))
        self.pCoord = pCoord[np.newaxis, :, :, :]
        if isCuda:
            self.cameraPos = self.cameraPos.cuda()
            self.pCoord = self.pCoord.cuda()

    def forward_batch(self, diffusePred, normalPred, roughPred, distPred, segBatch, lightPos):
        batch_size = diffusePred.size(0)
        res = []
        for i in range(batch_size):
            res += [self.forward(diffusePred[i].unsqueeze(0), normalPred[i].unsqueeze(0), \
                                 roughPred[i].unsqueeze(0), distPred[i].unsqueeze(0), \
                                 segBatch[i].unsqueeze(0), lightPos[i].unsqueeze(0))]
        return torch.cat(res, 0)

    def forward(self, diffusePred, normalPred, roughPred, distPred, segBatch, lightPos, gpuId=0):
        lightPos = lightPos.view(1, 3, 1, 1)
        v = self.cameraPos - self.pCoord
        l = lightPos - self.pCoord
        v = (v / torch.sqrt(torch.clamp(torch.sum(v*v, dim=1), 1e-12).unsqueeze(1))).float()
        l = (l / torch.sqrt(torch.clamp(torch.sum(l*l, dim=1), 1e-12).unsqueeze(1))).float()

        h = (v + l) / 2
        h = (h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=1), 1e-12).unsqueeze(1))).float()

        temp = torch.FloatTensor(1, 1, 1, 1)
        pCoord = self.pCoord

        if self.isCuda:
            v = v.cuda(gpuId)
            l = l.cuda(gpuId)
            h = h.cuda(gpuId)
            pCoord = pCoord.cuda(gpuId)
            temp = temp.cuda(gpuId)
            lightPos = lightPos.cuda(gpuId)

        vdh = torch.sum((v * h), dim = 1)
        vdh = vdh.unsqueeze(0)
        temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        # render
        diffuseBatch = (diffusePred + 1.0)/2.0 / np.pi
        roughBatch = (roughPred + 1.0)/2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normalPred * v.expand_as(normalPred), dim=1), 0, 1)
        ndh = torch.clamp(torch.sum(normalPred * h.expand_as(normalPred), dim=1), 0, 1)
        ndl = torch.clamp(torch.sum(normalPred * l.expand_as(normalPred), dim=1), 0, 1)

        if len(ndv.size()) == 3:
            ndv = ndv.unsqueeze(1)
            ndh = ndh.unsqueeze(1)
            ndl = ndl.unsqueeze(1)

        frac = alpha2 * frac0.expand_as(alpha)
        nom0 = ndh * ndh * (alpha2 - 1) + 1  # D
        nom1 = ndv * (1 - k) + k  # G
        nom2 = ndl * (1 - k) + k  # G
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom

        coord3D = pCoord.expand_as(diffuseBatch) * distPred.expand_as(diffuseBatch)
        dist2Pred = torch.sum( (lightPos.expand_as(coord3D) - coord3D) \
                * (lightPos.expand_as(coord3D) - coord3D), dim=1).unsqueeze(1)
        color = (diffuseBatch + specPred.expand_as(diffusePred)) * ndl.expand_as(diffusePred) * \
                self.lightPower / torch.clamp(dist2Pred.expand_as(diffusePred), 1e-6)
        color = color * segBatch.expand_as(diffusePred)
        return torch.clamp(color, 0, 1)


class RenderLayerDirecLightTorch():
    def __init__(self, imSize=256, fov=60, F0=0.05, cameraPos=[0, 0, 0],
            lightPower=1, isCuda=True):
        self.imSize = imSize
        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = torch.from_numpy(np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1]))
        self.yRange = self.xRange = 1 * np.tan(self.fov/2)
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imSize),
                np.linspace(-self.yRange, self.yRange, imSize) )
        y = np.flip(y, axis=0)
        z = -np.ones((imSize, imSize), dtype=np.float32)

        pCoord = torch.from_numpy(np.stack([x, y, z]).astype(np.float32))
        self.pCoord = pCoord[np.newaxis, :, :, :]
        if isCuda:
            self.cameraPos = self.cameraPos.cuda()
            self.pCoord = self.pCoord.cuda()

    def forward_batch(self, diffusePred, normalPred, roughPred, distPred, segBatch, lightDirec):
        batch_size = diffusePred.size(0)
        res = []
        for i in range(batch_size):
            res += [self.forward(diffusePred[i].unsqueeze(0), normalPred[i].unsqueeze(0), \
                                    roughPred[i].unsqueeze(0), distPred[i].unsqueeze(0),  \
                                    segBatch[i].unsqueeze(0), lightDirec[i].unsqueeze(0))]
        return torch.cat(res, 0)

    def forward(self, diffusePred, normalPred, roughPred, distPred, segBatch, lightDirec, gpuId=0):
        lightDirec = lightDirec.view(1, 3, 1, 1)
        v = self.cameraPos - self.pCoord
        l = lightDirec
        v = (v / torch.sqrt(torch.clamp(torch.sum(v*v, dim=1), 1e-12).unsqueeze(1))).float()
        l = (l / torch.sqrt(torch.clamp(torch.sum(l*l, dim=1), 1e-12).unsqueeze(1))).float()

        h = (v + l) / 2
        h = (h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=1), 1e-12).unsqueeze(1))).float()

        temp = torch.FloatTensor(1, 1, 1, 1)
        pCoord = self.pCoord
        
        if self.isCuda:
            v = v.cuda(gpuId)
            l = l.cuda(gpuId)
            h = h.cuda(gpuId)
            pCoord = pCoord.cuda(gpuId)
            temp = temp.cuda(gpuId)

        vdh = torch.sum((v * h), dim = 1)
        vdh = vdh.unsqueeze(0)
        temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        # render
        diffuseBatch = (diffusePred + 1.0)/2.0 / np.pi
        roughBatch = (roughPred + 1.0)/2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normalPred * v.expand_as(normalPred), dim=1), 0, 1)
        ndh = torch.clamp(torch.sum(normalPred * h.expand_as(normalPred), dim=1), 0, 1)
        ndl = torch.clamp(torch.sum(normalPred * l.expand_as(normalPred), dim=1), 0, 1)

        if len(ndv.size()) == 3:
            ndv = ndv.unsqueeze(1)
            ndh = ndh.unsqueeze(1)
            ndl = ndl.unsqueeze(1)

        frac = alpha2 * frac0.expand_as(alpha)
        nom0 = ndh * ndh * (alpha2 - 1) + 1  # D
        nom1 = ndv * (1 - k) + k  # G
        nom2 = ndl * (1 - k) + k  # G
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom

        coord3D = pCoord.expand_as(diffuseBatch) * distPred.expand_as(diffuseBatch)
        color = (diffuseBatch + specPred.expand_as(diffusePred)) * ndl.expand_as(diffusePred)
                
        color = color * segBatch.expand_as(diffusePred)
        return torch.clamp(color, 0, 1)


class RenderLayerPointLightEnvTorch():
    def __init__(self, imSize=256, fov=60, F0=0.05, cameraPos=[0, 0, 0], 
            lightPower=5.95, isCuda=True):
        self.imSize = imSize
        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = torch.from_numpy(np.array(cameraPos, \
                                dtype=np.float32).reshape([1, 3, 1, 1]))
        self.lightPower = lightPower
        self.yRange = self.xRange = 1 * np.tan(self.fov/2)
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imSize),
                np.linspace(-self.yRange, self.yRange, imSize) )
        y = np.flip(y, axis=0)
        z = -np.ones((imSize, imSize), dtype=np.float32)

        pCoord = torch.from_numpy(np.stack([x, y, z]).astype(np.float32))
        self.pCoord = pCoord[np.newaxis, :, :, :]
        if isCuda:
            self.cameraPos = self.cameraPos.cuda()
            self.pCoord = self.pCoord.cuda()

    def forward_batch(self, diffusePred, normalPred, roughPred, distPred, segBatch, lightPos):
        batch_size = diffusePred.size(0)
        res = []
        for i in range(batch_size):
            res += [self.forward(diffusePred[i].unsqueeze(0), normalPred[i].unsqueeze(0), \
                                 roughPred[i].unsqueeze(0), distPred[i].unsqueeze(0), \
                                 segBatch[i].unsqueeze(0), lightPos[i].unsqueeze(0))]
        return torch.cat(res, 0)

    def forward(self, diffusePred, normalPred, roughPred, distPred, segBatch, lightPos, gpuId=0):
        lightPos = lightPos.view(1, 3, 1, 1)
        v = self.cameraPos - self.pCoord
        l = lightPos - self.pCoord
        v = (v / torch.sqrt(torch.clamp(torch.sum(v*v, dim=1), 1e-12).unsqueeze(1))).float()
        l = (l / torch.sqrt(torch.clamp(torch.sum(l*l, dim=1), 1e-12).unsqueeze(1))).float()

        h = (v + l) / 2
        h = (h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=1), 1e-12).unsqueeze(1))).float()

        temp = torch.FloatTensor(1, 1, 1, 1)
        pCoord = self.pCoord

        if self.isCuda:
            v = v.cuda(gpuId)
            l = l.cuda(gpuId)
            h = h.cuda(gpuId)
            pCoord = pCoord.cuda(gpuId)
            temp = temp.cuda(gpuId)
            lightPos = lightPos.cuda(gpuId)

        vdh = torch.sum((v * h), dim = 1)
        vdh = vdh.unsqueeze(0)
        temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        # render
        diffuseBatch = (diffusePred + 1.0)/2.0 / np.pi
        roughBatch = (roughPred + 1.0)/2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normalPred * v.expand_as(normalPred), dim=1), 0, 1)
        ndh = torch.clamp(torch.sum(normalPred * h.expand_as(normalPred), dim=1), 0, 1)
        ndl = torch.clamp(torch.sum(normalPred * l.expand_as(normalPred), dim=1), 0, 1)

        if len(ndv.size()) == 3:
            ndv = ndv.unsqueeze(1)
            ndh = ndh.unsqueeze(1)
            ndl = ndl.unsqueeze(1)

        frac = alpha2 * frac0.expand_as(alpha)
        nom0 = ndh * ndh * (alpha2 - 1) + 1  # D
        nom1 = ndv * (1 - k) + k  # G
        nom2 = ndl * (1 - k) + k  # G
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom

        coord3D = pCoord.expand_as(diffuseBatch) * distPred.expand_as(diffuseBatch)
        dist2Pred = torch.sum( (lightPos.expand_as(coord3D) - coord3D) \
                * (lightPos.expand_as(coord3D) - coord3D), dim=1).unsqueeze(1)
        color = (diffuseBatch + specPred.expand_as(diffusePred)) * ndl.expand_as(diffusePred) * \
                self.lightPower / torch.clamp(dist2Pred.expand_as(diffusePred), 1e-6)
        color = color * segBatch.expand_as(diffusePred)
        return torch.clamp(color, 0, 1)


    def forward_env(self, diffusePred, normalPred, roughPred, segBatch, SHPred):
        diffuseBatch = (diffusePred + 1) / 2.0 / np.pi
        batchSize = diffusePred.size(0)
        c1, c2, c3, c4, c5 = 0.429043, 0.511664, 0.743125, 0.886227, 0.247708
        L0, L1_1, L10, L11, L2_2, L2_1, L20, L21, L22 = torch.split(SHPred, 1, dim=2)
        nx, ny, nz = torch.split(normalPred, 1, dim=1)
        L0, L1_1, L10, L11 = L0.contiguous(), L1_1.contiguous(), L10.contiguous(), L11.contiguous()
        L2_2, L2_1, L20, L21, L22 = L2_2.contiguous(), L2_1.contiguous(), \
                L20.contiguous(), L21.contiguous(), L22.contiguous()
        L0 = L0.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize] )
        L1_1 = -L1_1.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        L10 = L10.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        L11 = -L11.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        L2_2 = L2_2.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        L2_1 = -L2_1.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        L20 = L20.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        L21 = -L21.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        L22 = L22.view(batchSize, 3, 1, 1).expand([batchSize, 3, self.imSize, self.imSize])
        nx = nx.expand([batchSize, 3, self.imSize, self.imSize])
        ny = ny.expand([batchSize, 3, self.imSize, self.imSize])
        nz = nz.expand([batchSize, 3, self.imSize, self.imSize])

        x = c1*L22*nx + c1*L2_2*ny + c1*L21*nz + c2*L11
        y = c1*L2_2*nx - c1*L22*ny + c1*L2_1*nz + c2*L1_1
        z = c1*L21*nx + c1*L2_1*ny + c3*L20*nz + c2*L10
        w = c2*L11*nx + c2*L1_1*ny + c2*L10*nz + c4*L0 - c5*L20

        radiance = nx*x + ny*y + nz*z + w

        color = diffuseBatch * radiance * segBatch.expand_as(diffusePred)
        return torch.clamp(color, 0, 1)
