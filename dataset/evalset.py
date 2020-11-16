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
            self.imBg    = self.next_batch['image_bg'].cuda(non_blocking=True)
            self.image_pt_src = self.next_batch['image_pt_src'].cuda(non_blocking=True)
            self.image_pt_tar = self.next_batch['image_pt_tar'].cuda(non_blocking=True)
            self.light_pt_tar = self.next_batch['light_pt_tar'].cuda(non_blocking=True)
            self.image_env_src = self.next_batch['image_env_src'].cuda(non_blocking=True)
            self.image_env_tar = self.next_batch['image_env_tar'].cuda(non_blocking=True)
            self.light_env_tar = self.next_batch['light_env_tar'].cuda(non_blocking=True)

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
        imBg   = self.imBg

        image_pt_src = self.image_pt_src
        image_pt_tar = self.image_pt_tar
        light_pt_tar = self.light_pt_tar

        image_env_src = self.image_env_src
        image_env_tar = self.image_env_tar
        light_env_tar = self.light_env_tar

        self.preload()

        return {'albedo' : albedo,
                'normal' : normal,
                'rough'  : rough,
                'depth'  : depth,
                'seg'    : seg,
                'SH'     : SH,
                'image_bg': imBg,
                'image_pt_src': image_pt_src,
                'image_pt_tar': image_pt_tar,
                'light_pt_tar': light_pt_tar,
                'image_env_src': image_env_src,
                'image_env_tar': image_env_tar,
                'light_env_tar': light_env_tar}


class SyntheticData(Dataset):
    def __init__(self, dataRoot, imSize=256, isRandom=True, phase='TEST', rseed=None):
        dataRoot = osp.join(dataRoot, phase.lower())
        print('--> data root: %s' % dataRoot)
        
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
            self.SHList     = pk_dict['SH_list']
            self.bgList     = pk_dict['bg_list']
            self.imagePtSrcList    = pk_dict['image_pt_src_list']
            self.imagePtTarList    = pk_dict['image_pt_tar_list']
            self.imagePtLightList  = pk_dict['light_pt_tar_list']
            self.imageEnvSrcList   = pk_dict['image_env_tar_list']
            self.imageEnvTarList   = pk_dict['image_env_tar_list']
            self.imageEnvLightList = pk_dict['light_env_tar_list']
        else:
            print('--> you are not using `pickle` for fast loading, \
                    please use `python dataset/make_pkl.py` to create one')
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
            # Geometry
            self.depthList = [x.replace('albedo', 'depth').replace('png', 'dat') for x in self.albedoList]

            # Bg
            self.bgList = [x.replace('albedo', 'imgEnv') for x in self.albedoList]

            # Environment Map
            self.SHList = []
            for x in self.albedoList:
                suffix = '/'.join(x.split('/')[0:-1])
                fileName = x.split('/')[-1]
                fileName = fileName.split('_')
                self.SHList.append(osp.join(suffix, '_'.join(fileName[0:2]) + '.npy'))

            # Rendering
            self.imagePtSrcList = []
            self.imagePtTarList = []
            self.imagePtLightList = []
            for x in self.albedoList:
                src_name, tar_name, tar_light = self.get_image_name(x, pt=True)
                self.imagePtSrcList.append(src_name)
                self.imagePtTarList.append(tar_name)
                self.imagePtLightList.append(tar_light)

            self.imageEnvSrcList = []
            self.imageEnvTarList = []
            self.imageEnvLightList = []
            for x in self.albedoList:
                src_name, tar_name, tar_light = self.get_image_name(x, pt=False)
                self.imageEnvSrcList.append(src_name)
                self.imageEnvTarList.append(tar_name)
                self.imageEnvLightList.append(tar_light)

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

        # Read depth
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

        # Read SH
        if not os.path.isfile(self.SHList[self.perm[ind]]):
            print('Fail to load {0}'.format(self.SHList[self.perm[ind]]))
            SH = np.zeros([3, 9], dtype=np.float32)
        else:
            SH = np.load(self.SHList[self.perm[ind]]).transpose([1, 0] )[:, 0:9]
            SH = SH.astype(np.float32)[::-1, :]

        # Read backgrounds
        imBg = self.loadImage(self.bgList[self.perm[ind]], isGama=True)

        # Scale the Environment
        scaleEnv = 0.5
        imBg = (((imBg + 1) * 0.5) * scaleEnv) * (1 - seg)
        SH = SH * scaleEnv

        image_pt_src = self.loadImage(self.imagePtSrcList[self.perm[ind]], isGama=True)
        image_pt_tar = self.loadImage(self.imagePtTarList[self.perm[ind]], isGama=True)
        light_pt_tar = self.imagePtLightList[self.perm[ind]]

        image_env_src = self.loadImage(self.imageEnvSrcList[self.perm[ind]], isGama=True)
        image_env_tar = self.loadImage(self.imageEnvTarList[self.perm[ind]], isGama=True)
        light_env_tar = self.imageEnvLightList[self.perm[ind]]

        batchDict = {'albedo' : albedo,
                     'normal' : normal,
                     'rough'  : rough,
                     'depth'  : depth,
                     'seg'    : seg,
                     'SH'     : SH,
                     'image_bg': imBg,
                     'image_pt_src': image_pt_src,
                     'image_pt_tar': image_pt_tar,
                     'light_pt_tar': light_pt_tar,
                     'image_env_src': image_env_src,
                     'image_env_tar': image_env_tar,
                     'light_env_tar': light_env_tar}
        return batchDict

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
        assert((w0 == h0))
        im = im.resize((self.imSize, self.imSize), Image.ANTIALIAS)
        return im
    
    def get_image_name(self, albedo_name, pt=True):
        name = albedo_name
        if pt:
            img_names = glob.glob(name.replace('albedo', 'image_pt*'))
        else:
            img_names = glob.glob(name.replace('albedo', 'image_env*'))

        assert len(img_names) == 2, print(albedo_name)
        lights_str = [n.replace('[', '\t').replace(']', '\t').split('\t')[1] for n in img_names]
        
        lights = []
        for l in lights_str:
            lights.append([float(i) for i in l.split(',')])
        assert ([0., 0., 0.] in lights), print(lights)

        if lights[0] == [0., 0., 0.]:
            img_src = img_names[0]
            img_tar = img_names[1]
            light_tar = lights[1]
        elif lights[1] == [0., 0., 0.]:
            img_src = img_names[1]
            img_tar = img_names[0]
            light_tar = lights[0]

        # print('[TEST]', img_src, img_tar, light_tar)
        return img_src, img_tar, np.array(light_tar, dtype=np.float32)