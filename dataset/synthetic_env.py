import os
import cv2
import glob
import torch
import pickle
import random
import struct
import numpy as np
import os.path as osp
import scipy.ndimage as ndimage

from PIL import Image
from torch.utils.data import Dataset


class DataPrefetcher():
    """For accelerating data loader
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            print('--> stop iteration')
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.albedo  = self.next_batch['albedo'].cuda(non_blocking=True)
            self.normal  = self.next_batch['normal'].cuda(non_blocking=True)
            self.rough   = self.next_batch['rough' ].cuda(non_blocking=True)
            self.depth   = self.next_batch['depth' ].cuda(non_blocking=True)
            self.seg     = self.next_batch['seg'   ].cuda(non_blocking=True)
            self.SH      = self.next_batch['SH'    ].cuda(non_blocking=True)
            self.image_bg = self.next_batch['image_bg'].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_batch is None:
            return

        albedo = self.albedo
        normal = self.normal
        rough  = self.rough
        depth  = self.depth
        seg    = self.seg
        SH     = self.SH
        image_bg = self.image_bg
        
        self.preload()

        return {'albedo':   albedo,
                'normal':   normal,
                'rough':    rough,
                'depth':    depth,
                'seg':      seg,
                'SH':       SH,
                'image_bg': image_bg}


class SyntheticData(Dataset):
    def __init__(self, dataRoot, imSize=256, isRandom=True, phase='TRAIN', rseed=0):
        dataRoot = osp.join(dataRoot, phase.lower())
        print('--> dara root: %s' % dataRoot)
        
        if not osp.exists(dataRoot):
            raise ValueError('Wrong data root!')

        self.dataRoot = dataRoot
        self.imSize = imSize

        shapeList = glob.glob(osp.join(dataRoot, 'Shape__*') )
        shapeList = sorted(shapeList)

        self.albedoList = []
        for shape in shapeList:
            albedoNames = glob.glob(osp.join(shape, '*albedo.png') )
            albedoNames = sorted(albedoNames)
            self.albedoList = self.albedoList + albedoNames

        if rseed is not None:
            random.seed(rseed)

        # BRDF parameter
        self.normalList = [x.replace('albedo', 'normal') for x in self.albedoList]
        self.roughList = [x.replace('albedo', 'rough') for x in self.albedoList]
        self.segList = [x.replace('albedo', 'seg') for x in self.albedoList]
        # Geometry
        self.depthList = [x.replace('albedo', 'depth').replace('png', 'dat') for x in self.albedoList]

        # Env
        self.imEList = [x.replace('albedo', 'imgEnv') for x in self.albedoList]

        # Environment Map
        self.SHList = []
        self.nameList = []
        for x in self.albedoList:
            suffix = '/'.join(x.split('/')[0:-1])
            fileName = x.split('/')[-1]
            fileName = fileName.split('_')
            self.SHList.append(osp.join(suffix, '_'.join(fileName[0:2]) + '.npy'))
            self.nameList.append(osp.join(suffix, '_'.join(fileName[0:3])))

        # Permute the image list
        self.count = len(self.albedoList)
        self.perm = list(range(self.count))
        if isRandom:
            random.shuffle(self.perm)


    def __len__(self):
        return len(self.perm)


    def __getitem__(self, ind):
        # Read segmentation
        seg = 0.5 * self.loadImage(self.segList[self.perm[ind]]) + 0.5
        seg = (seg[0, :, :] > 0.999999).astype(dtype = np.int)
        seg = ndimage.binary_erosion(seg, structure = np.ones((2, 2))).astype(dtype=np.float32)
        seg = seg[np.newaxis, :, :]

        # Read albedo
        albedo = self.loadImage(self.albedoList[self.perm[ind]])
        albedo = albedo * seg

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(self.normalList[self.perm[ind]])
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :]
        normal = normal * seg

        # Read roughness
        rough = self.loadImage(self.roughList[self.perm[ind]])[0:1, :, :]
        rough = (rough * seg)

        # Read rendered images
        imE = self.loadImage(self.imEList[self.perm[ind]], isGama=True)
        imEbg = imE.copy()

        with open(self.depthList[self.perm[ind]], 'rb') as f:
            byte = f.read()
            if len(byte) == 256 * 256 * 3 * 4:
                depth = np.array(struct.unpack(str(256*256*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([256, 256, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1])
                depth = depth * seg
            elif len(byte) == 512 * 512 * 3 * 4:
                print(self.depthList[self.perm[ind]])
                assert(False)

        if not os.path.isfile(self.SHList[self.perm[ind]]):
            print('Fail to load {0}'.format(self.SHList[self.perm[ind]]))
            SH = np.zeros([3, 9], dtype=np.float32)
        else:
            SH = np.load(self.SHList[self.perm[ind]]).transpose([1, 0] )[:, 0:9]
            SH = SH.astype(np.float32)[::-1, :]
        name = self.nameList[self.perm[ind]]

        # Scale the Environment
        scaleEnv = 0.5
        imEbg = ((imEbg + 1) * scaleEnv - 1)
        imEbg = ((imEbg + 1) * 0.5) * (1 - seg)
        SH = SH * scaleEnv

        batchDict = {'albedo':   albedo,
                     'normal':   normal,
                     'rough':    rough,
                     'depth':    depth,
                     'seg':      seg,
                     'SH':       SH,
                     'image_bg': image_bg}

        return batchDict


    def loadImage(self, imName, isGama = False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            return im

        im = Image.open(imName)
        im = self.imResize(im)
        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        return im

    def imResize(self, im):
        w0, h0 = im.size
        assert( (w0 == h0) )
        im = im.resize((self.imSize, self.imSize), Image.ANTIALIAS)
        return im
