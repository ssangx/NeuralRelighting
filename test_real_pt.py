import os
import cv2
import glob
import torch
import struct
import numpy as np
import os.path as osp
import torch.nn as nn
import scipy.ndimage as ndimage
import torchvision.utils as vutils

from options import test_options
from models import network, render

from PIL import Image

# Options
opts = test_options.TestOptions().parse()
opts.name = 'pt'
opts.outf = './data/output'
opts.workers = 2
opts.batch_size = 1
opts.gpu_id = [0]

print('--> pytorch can use %d GPUs'  % (torch.cuda.device_count()))
print('--> pytorch is using %d GPUs' % (len(opts.gpu_id)))
print('--> GPU IDs:', opts.gpu_id)

# Model 
encoder = nn.DataParallel(network.encoderInitial(4), device_ids=opts.gpu_id).cuda()
decoder_brdf = nn.DataParallel(network.decoderBRDF(), device_ids=opts.gpu_id).cuda()
decoder_render = nn.DataParallel(network.decoderRender(), device_ids=opts.gpu_id).cuda()

render_layer = render.RenderLayerPointLightTorch()

encoderRef = nn.DataParallel(network.RefineEncoder(), device_ids=opts.gpu_id).cuda()
decoderRef_brdf = nn.DataParallel(network.RefineDecoderBRDF(), device_ids=opts.gpu_id).cuda()
decoderRef_render = nn.DataParallel(network.RefineDecoderRender(), device_ids=opts.gpu_id).cuda()

encoderRef2 = nn.DataParallel(network.RefineEncoder(), device_ids=opts.gpu_id).cuda()
decoderRef2_brdf = nn.DataParallel(network.RefineDecoderBRDF(), device_ids=opts.gpu_id).cuda()
decoderRef2_render = nn.DataParallel(network.RefineDecoderRender(), device_ids=opts.gpu_id).cuda()

path = './data/models/%s' % (opts.name)
encoder.load_state_dict(torch.load('%s/encoder.pth'  % path, map_location=lambda storage, loc:storage))
decoder_brdf.load_state_dict(torch.load('%s/decoder_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoder_render.load_state_dict(torch.load('%s/decoder_render.pth' % path, map_location=lambda storage, loc:storage))

encoderRef.load_state_dict(torch.load('%s/encoderRef.pth' % path, map_location=lambda storage, loc:storage))
decoderRef_brdf.load_state_dict(torch.load('%s/decoderRef_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoderRef_render.load_state_dict(torch.load('%s/decoderRef_render.pth' % path, map_location=lambda storage, loc:storage))

encoderRef2.load_state_dict(torch.load('%s/encoderRef2.pth' % path, map_location=lambda storage, loc:storage))
decoderRef2_brdf.load_state_dict(torch.load('%s/decoderRef2_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoderRef2_render.load_state_dict(torch.load('%s/decoderRef2_render.pth' % path, map_location=lambda storage, loc:storage))

def _fix(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

_fix(encoder)
_fix(decoder_brdf)
_fix(decoder_render)

_fix(encoderRef)
_fix(decoderRef_brdf)
_fix(decoderRef_render)

_fix(encoderRef2)
_fix(decoderRef2_brdf)
_fix(decoderRef2_render)

angle = 45

save_path = '%s/%s/show_%d/' % (opts.outf, opts.name, angle)
if not os.path.exists(save_path):
    os.makedirs(save_path)

test_list = glob.glob(osp.join('./data/real/pt', 'test*'))
image_list = []
for case in test_list:
    image = glob.glob(osp.join(case, '*image*.png'))
    image_list += image
seg_list = [x.replace('image', 'mask') for x in image_list]

print('# of test:', len(test_list))

def gen_sliding_lights(samples=120):
    y = 0
    lights = []
    for i in range(samples):
        phi = - np.pi/3 + 2*np.pi/3 / (samples-1) * i
        z = np.cos(phi) - 1
        x = np.sin(phi)
        lights += [torch.from_numpy(np.array([[x, y, z]])).float().cuda()]
    return lights

def gen_circular_lights(theta=np.pi * angle / 180, samples=120):
    z = np.sin(theta) - 1
    r = np.cos(theta)
    lights = []
    for i in range(samples):
        phi = np.pi * 2 / (samples-1) * i
        x = np.sin(phi) * r
        y = np.cos(phi) * r
        lights += [torch.from_numpy(np.array([[x, y, z]])).float().cuda()]
    return lights


def writeImageToFile(imgBatch, nameBatch):
    batchSize = imgBatch.size(0)
    assert batchSize == 1
    for n in range(0, batchSize):
        img = imgBatch[n, :, :, :].data.cpu().numpy()
        img = np.clip(img, 0, 1)
        img = (255 * img.transpose([1, 2, 0])).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img )
        img.save(nameBatch)

def loadImage(imName, isGama=False):
    im = Image.open(imName)
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


for _i in range(len(seg_list)):

    img_name = image_list[_i]
    seg_name = seg_list[_i]

    seg = np.array(Image.open(seg_name), dtype=np.float32) 
    seg = cv2.resize(seg, (256, 256))
    seg = (seg - 127.5) / 127.5
    seg = np.transpose(seg, [2, 0, 1])
    seg = 0.5 * seg + 0.5
    seg = (seg[0, :, :] > 0.999999).astype(dtype=np.int)
    seg = ndimage.binary_erosion(seg, structure=np.ones((2, 2))).astype(dtype=np.float32)
    seg = seg[np.newaxis, :, :]

    img = np.array(Image.open(img_name), dtype=np.float32)[:, :, :3]
    img = cv2.resize(img, (256, 256))
    img = (img / 255.0) ** 2.2
    img = 2 * img - 1
    img = np.transpose(img, [2, 0, 1])


    seg   = torch.from_numpy(seg).cuda().unsqueeze(0)

    image_s = torch.from_numpy(img).cuda().unsqueeze(0) * seg
    light_s = torch.zeros(seg.size(0), 3).float().cuda().clamp(0, 1)

    if angle == 0:
        light_t = gen_sliding_lights()
    else:
        light_t = gen_circular_lights()
    
    # light_t += [light_s]

    input = torch.cat([image_s, seg], 1)
    init_feat = encoder(input)
    init_brdf_feat, init_brdf_pred = decoder_brdf(init_feat)
    albedo_pred, normal_pred, rough_pred, depth_pred = init_brdf_pred

    relit_pred_list = []
    relit_predcas_list = []
    relit_predcas2_list = []
    for t in light_t:
        relit_render = 2 * render_layer.forward_batch(albedo_pred, normal_pred, rough_pred, depth_pred, seg, t) - 1
        relit_pred = decoder_render(init_feat, init_brdf_feat, t)
        
        cas_input = torch.cat([image_s * seg, \
                                relit_pred * seg, \
                                relit_render * seg, \
                                albedo_pred * seg, \
                                normal_pred * seg, \
                                rough_pred * seg, \
                                depth_pred * seg, \
                                seg], dim=1)
        feat_cas = encoderRef(cas_input)
        brdf_feat_cas, brdf_pred_cas = decoderRef_brdf(feat_cas)
        albedo_pred_cas, normal_pred_cas, \
        rough_pred_cas, depth_pred_cas = brdf_pred_cas

        relit_casrender = 2 * render_layer.forward_batch(albedo_pred_cas, normal_pred_cas, rough_pred_cas, depth_pred_cas, seg, t) - 1
        relit_caspred = decoderRef_render(feat_cas, brdf_feat_cas, t)

        cas2_input = torch.cat([image_s * seg, \
                            relit_caspred * seg, \
                            relit_casrender * seg, \
                            albedo_pred_cas * seg, \
                            normal_pred_cas * seg, \
                            rough_pred_cas * seg, \
                            depth_pred_cas * seg, \
                            seg], dim=1)

        feat_cas2 = encoderRef2(cas2_input)
        brdf_feat_cas2, brdf_pred_cas2 = decoderRef2_brdf(feat_cas2)
        albedo_pred_cas2, normal_pred_cas2, \
        rough_pred_cas2, depth_pred_cas2 = brdf_pred_cas2

        relit_cas2render = 2 * render_layer.forward_batch(albedo_pred_cas2, normal_pred_cas2, rough_pred_cas2, depth_pred_cas2, seg, t) - 1
        relit_cas2pred = decoderRef2_render(feat_cas2, brdf_feat_cas2, t)

        relit_pred_list += [relit_pred]
        relit_predcas_list += [relit_caspred]
        relit_predcas2_list += [relit_cas2pred]

    # relit_pred_list = relit_pred_list[:-1]
    # relit_predcas_list = relit_predcas_list[:-1]
    # relit_predcas2_list = relit_predcas2_list[:-1]

    # ---------------- save BRDF reconstruction ------------------------
    albedo_save = 0.5 * (albedo_pred + 1) * seg
    normal_save = 0.5 * (normal_pred + 1) * seg
    rough_save  = 0.5 * (rough_pred  + 1) * seg
    rough_save = rough_save.expand_as(albedo_pred)
    depth_save  = 1 / torch.clamp(depth_pred, 1e-6, 10) * seg
    depth_save  = (depth_save - 0.25) / 0.8
    depth_save = depth_save.expand_as(albedo_save)
    brdf_save  = torch.cat([albedo_save, normal_save, rough_save, depth_save], dim=3)

    albedo_save_cas = 0.5 * (albedo_pred_cas + 1) * seg
    normal_save_cas = 0.5 * (normal_pred_cas + 1) * seg
    rough_save_cas  = 0.5 * (rough_pred_cas  + 1) * seg
    rough_save_cas = rough_save_cas.expand_as(albedo_save_cas)
    depth_save_cas  = 1 / torch.clamp(depth_pred_cas, 1e-6, 10) * seg
    depth_save_cas  = (depth_save_cas - 0.25) / 0.8
    depth_save_cas = depth_save_cas.expand_as(albedo_save_cas)
    brdf_save_cas = torch.cat([albedo_save_cas, normal_save_cas, rough_save_cas, depth_save_cas], dim=3)

    albedo_save_cas2 = 0.5 * (albedo_pred_cas2 + 1) * seg
    normal_save_cas2 = 0.5 * (normal_pred_cas2 + 1) * seg
    rough_save_cas2  = 0.5 * (rough_pred_cas2  + 1) * seg
    rough_save_cas2 = rough_save_cas2.expand_as(albedo_save_cas2)
    depth_save_cas2  = 1 / torch.clamp(depth_pred_cas2, 1e-6, 10) * seg
    depth_save_cas2  = (depth_save_cas2 - 0.25) / 0.8
    depth_save_cas2 = depth_save_cas2.expand_as(albedo_save_cas2)
    brdf_save_cas2 = torch.cat([albedo_save_cas2, normal_save_cas2, rough_save_cas2, depth_save_cas2], dim=3)
    
    brdf_together = torch.cat([brdf_save, brdf_save_cas, brdf_save_cas2], dim=2)

    path = osp.join(save_path, 'case%d' % _i)
    if not os.path.exists(path):
        os.mkdir(path)
    writeImageToFile(brdf_save,      path + '/brdf_0.png')
    writeImageToFile(brdf_save_cas,  path + '/brdf_1.png')
    writeImageToFile(brdf_save_cas2, path + '/brdf_2.png')
    writeImageToFile(brdf_together,  path + '/brdf.png')
    # ---------------------------------------------------------------------

    relit_preds = []
    relit_caspreds = []
    relit_cas2preds = []
    for img in relit_pred_list:
        relit_preds     += [((0.5 * (1 + img)) ** (1/2.2)).clamp_(0, 1) * seg]
    for img in relit_predcas_list:
        relit_caspreds  += [((0.5 * (1 + img)) ** (1/2.2)).clamp_(0, 1) * seg]
    for img in relit_predcas2_list:
        relit_cas2preds += [((0.5 * (1 + img)) ** (1/2.2)).clamp_(0, 1) * seg]

    image_src = ((0.5 * (1 + image_s)) ** (1/2.2)).clamp_(0, 1).clone() * seg
    writeImageToFile(image_src, path + '/image_src.png')

    for i in range(len(light_t)-1):
        writeImageToFile(relit_preds[i].clone(),     path + '/image_c0_%d.png' % (i))
        writeImageToFile(relit_caspreds[i].clone(),  path + '/image_c1_%d.png' % (i))
        writeImageToFile(relit_cas2preds[i].clone(), path + '/image_c2_%d.png' % (i))
    
    os.system('ffmpeg -framerate 24 -i {}/image_c0_\%d.png -c:v libx264 -loglevel panic -pix_fmt yuv420p {}/video_c0.mp4'.format(
                path, path))
    os.system('ffmpeg -framerate 24 -i {}/image_c1_\%d.png -c:v libx264 -loglevel panic -pix_fmt yuv420p {}/video_c1.mp4'.format(
                path, path))
    os.system('ffmpeg -framerate 24 -i {}/image_c2_\%d.png -c:v libx264 -loglevel panic -pix_fmt yuv420p {}/video_c2.mp4'.format(
                path, path))