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
            self.image   = self.next_batch['image_c'].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_batch is None:
            return

        albedo = self.albedo
        normal = self.normal
        rough  = self.rough
        depth  = self.depth
        seg    = self.seg

        image  = self.image
        
        self.preload()

        return {'albedo':  albedo,
                'normal':  normal,
                'rough':   rough,
                'depth':   depth,
                'seg':     seg,
                'image_c': image}


class SyntheticData(Dataset):
    def __init__(self, dataRoot, imSize=256, isRandom=True, phase='TRAIN', rseed=None):
        dataRoot = osp.join(dataRoot, phase.lower())
        print('--> dara root: %s' % dataRoot)
        
        if not osp.exists(dataRoot):
            raise ValueError('Wrong data root!')

        self.dataRoot = dataRoot
        self.imSize = imSize

        pk_path = dataRoot + '_file_list.pickle'
        print(pk_path)
        if os.path.exists(pk_path):
            print('--> load file list from pickle: %s' % pk_path)
            with open(pk_path, 'rb') as handle:
                pk_dict = pickle.load(handle)
            self.albedoList = pk_dict['albedo_list']
            self.normalList = pk_dict['normal_list']
            self.roughList  = pk_dict['rough_list']
            self.depthList  = pk_dict['depth_list']
            self.segList    = pk_dict['seg_list']
            self.images     = pk_dict['images_list']
            self.lights     = pk_dict['lights_list']
        else:
            print('--> you are not using `pickle` for fast loading, \
                    please use `python dataset/make_pickle.py` to create one')
            shapeList = glob.glob(osp.join(dataRoot, 'Shape__*') )
            shapeList = sorted(shapeList)

            self.albedoList = []
            for shape in shapeList:
                albedoNames = glob.glob(osp.join(shape, '*albedo.png') )
                self.albedoList = self.albedoList + albedoNames

            if rseed is not None:
                random.seed(rseed)

            # BRDF parameter
            self.normalList = [x.replace('albedo', 'normal') for x in self.albedoList]
            self.roughList = [x.replace('albedo', 'rough') for x in self.albedoList]
            self.segList = [x.replace('albedo', 'seg') for x in self.albedoList]

            # Rendered Image
            self.imPList = [x.replace('albedo', 'imgPoint') for x in self.albedoList]

            # Geometry
            self.depthList = [x.replace('albedo', 'depth').replace('png', 'dat') for x in self.albedoList]

            # All images
            self.images = []
            self.lights = []
            for x in self.albedoList:
                img_names, lights = self.get_image_name(x)
                self.images.append(img_names)
                self.lights.append(lights)

        # Permute the image list
        self.count = len(self.albedoList)
        self.perm = list(range(self.count))
        if isRandom:
            random.shuffle(self.perm)


    def __len__(self):
        return len(self.perm)


    def __getitem__(self, ind):
        idx = self.perm[ind]
        # Read segmentation
        seg = 0.5 * self.loadImage(self.segList[idx]) + 0.5
        seg = (seg[0, :, :] > 0.999999).astype(dtype = np.int)
        seg = ndimage.binary_erosion(seg, structure = np.ones((2, 2))).astype(dtype = np.float32)
        seg = seg[np.newaxis, :, :]

        # Read albedo
        albedo = self.loadImage(self.albedoList[idx])
        albedo = albedo * seg

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(self.normalList[idx])
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
        normal = normal * seg

        # Read roughness
        rough = self.loadImage(self.roughList[idx])[0:1, :, :]
        rough = (rough * seg)

        with open(self.depthList[idx], 'rb') as f:
            byte = f.read()
            if len(byte) == 256 * 256 * 3 * 4:
                depth = np.array(struct.unpack(str(256*256*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([256, 256, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1])
                if self.imSize == 128:
                    depth = depth[:, ::2, ::2]
                elif self.imSize == 64:
                    depth = depth[:, ::4, ::4]
                depth = depth * seg
            elif len(byte) == 512 * 512 * 3 * 4:
                depth = np.array(struct.unpack(str(512*512*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([512, 512, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1])
                if self.imSize == 128:
                    depth = depth[:, ::4, ::4]
                elif self.imSize == 64:
                    depth = depth[:, ::8, ::8]
                depth = depth * seg

        assert ([0., 0., 0.] in self.lights[idx])
        for img, lit in zip(self.images[idx], self.lights[idx]):
            if lit == [0., 0., 0.]:
                image_name = img
        image = self.loadImage(image_name, isGama=True) * seg
        image = np.clip(image, -1, 1)

        batchDict = {'albedo' : albedo,
                     'normal' : normal,
                     'rough'  : rough,
                     'depth'  : depth,
                     'seg'    : seg,
                     'image_c': image}
        return batchDict


    def get_image_name(self, albedo_name):
        name = albedo_name
        img_names = glob.glob(name.replace('albedo', 'image_*'))
        assert len(img_names) == 3
        lights_str = [n.replace('[', '\t').replace(']', '\t').split('\t')[1] for n in img_names]
        lights = []
        for l in lights_str:
            lights.append([float(i) for i in l.split()])
        return img_names, lights

    def loadImage(self, imName, isGama=False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName))
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

    def loadNpy(self, name):
        data = np.load(name)
        return data