import glob
import torch
import struct
import numpy as np
import os.path as osp
import scipy.ndimage as ndimage

from PIL import Image
from utils import light
from models import renderer
from torch.utils.data import Dataset


class DataRender(Dataset):
    def __init__(self, dataRoot, imSize=256):
        self.imSize = imSize

        if not osp.exists(dataRoot):
            raise ValueError('Invalid data root!')

        shapeList = glob.glob(osp.join(dataRoot, 'Shape__*'))
        shapeList = sorted(shapeList)

        self.albedoList = []
        for shape in shapeList:
            albedos = glob.glob(osp.join(shape, '*albedo.png'))
            self.albedoList = self.albedoList + albedos

        # BRDF parameter
        self.normalList = [x.replace('albedo', 'normal') for x in self.albedoList]
        self.roughList = [x.replace('albedo', 'rough') for x in self.albedoList]
        self.depthList = [x.replace('albedo', 'depth').replace('png', 'dat') for x in self.albedoList]
        self.segList = [x.replace('albedo', 'seg') for x in self.albedoList]

        self.perm = list(range(len(self.segList)))

    def __len__(self):
        return len(self.segList)

    def __getitem__(self, idx):
        name = self.albedoList[idx]

        # Read segmentation
        mask = 0.5 * self.loadImage(self.segList[idx]) + 0.5
        mask = (mask[0, :, :] > 0.999999).astype(dtype = np.int)
        mask = ndimage.binary_erosion(mask, structure = np.ones( (2, 2))).astype(dtype = np.float32)
        mask = mask[np.newaxis, :, :]

        # Read albedo
        albedo = self.loadImage(self.albedoList[idx])
        albedo = albedo * mask

        # Read normal
        normal = self.loadImage(self.normalList[idx])
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
        normal = normal * mask

        # Read roughness
        rough = self.loadImage(self.roughList[idx])[0:1, :, :]
        rough = (rough * mask)

        # Read depth
        with open(self.depthList[idx ], 'rb') as f:
            byte = f.read()
            if len(byte) == 256 * 256 * 3 * 4:
                depth = np.array(struct.unpack(str(256*256*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([256, 256, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1])
                if self.imSize == 128:
                    depth = depth[:, ::2, ::2]
                elif self.imSize == 64:
                    depth = depth[:, ::4, ::4]
                depth = depth * mask
            elif len(byte) == 512 * 512 * 3 * 4:
                depth = np.array(struct.unpack(str(512*512*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([512, 512, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1])
                if self.imSize == 128:
                    depth = depth[:, ::4, ::4]
                elif self.imSize == 64:
                    depth = depth[:, ::8, ::8]
                depth = depth * mask

        batchDict = {'albedo': albedo,
                     'normal': normal,
                     'rough' : rough,
                     'depth' : depth,
                     'mask'  : mask,
                     'name'  : name}
        return batchDict

    def loadImage(self, imName, isGama=False):
        if not osp.isfile(imName):
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


def render_one(render_layer, name, albedo, normal, rough, depth, mask, light):
    image = render_layer.forward(albedo, normal, rough, depth, mask, light) ** (1/2.2)
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    fn = name.replace('albedo', 'image_{}'.format(list(light.cpu().numpy())))
    print('[RENDERING] rendering and saving {}'.format(fn))
    Image.fromarray(image).save(fn)


def rendering(root, cnt=1):
    render_layer = renderer.RenderLayerPointLightTorch()

    dataRender = DataRender(root)
    for data in dataRender:
        name = data['name']

        albedo = torch.FloatTensor(data['albedo']).cuda(non_blocking=True).unsqueeze(0)
        normal = torch.FloatTensor(data['normal']).cuda(non_blocking=True).unsqueeze(0)
        rough = torch.FloatTensor(data['rough']).cuda(non_blocking=True).unsqueeze(0)
        depth = torch.FloatTensor(data['depth']).cuda(non_blocking=True).unsqueeze(0)
        mask = torch.FloatTensor(data['mask']).cuda(non_blocking=True).unsqueeze(0)

        # render the input image
        light_c = torch.zeros(3).float().cuda()
        render_one(render_layer, name, albedo, normal, rough, depth, mask, light_c)

        # render relighting targets
        for _ in range(cnt):
            light_t = light.gen_uniform_in_hemisphere()
            render_one(render_layer, name, albedo, normal, rough, depth, mask, light_t)


if __name__ == "__main__":
    # TODO: modify it if you are using a different root
    root = './data/dataset/Synthetic'
    rendering(root=root, cnt=1)