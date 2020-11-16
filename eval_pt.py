import os
import cv2
import time
import torch
import numpy as np
import os.path as osp
import torch.nn as nn

from utils import logger
from dataset import evalset
from options import test_options
from models import network, renderer

from utils.ssim import SSIM, MSSSIM


# Options
opts = test_options.TestOptions().parse()
opts.outf = './data'
opts.gpu_id = [0]
opts.workers = 4
opts.batch_size = 1
opts.data_root = './data/dataset/Synthetic/'

logger.print_options(opts)

print('--> pytorch can use %d GPUs'  % (torch.cuda.device_count()))
print('--> pytorch is using %d GPUs' % (len(opts.gpu_id)))
print('--> GPU IDs:', opts.gpu_id)

# Model 
encoder = nn.DataParallel(network.encoderInitial(4), device_ids=opts.gpu_id).cuda()
decoder_brdf = nn.DataParallel(network.decoderBRDF(), device_ids=opts.gpu_id).cuda()
decoder_render = nn.DataParallel(network.decoderRender(), device_ids=opts.gpu_id).cuda()

encoderRef = nn.DataParallel(network.RefineEncoder(), device_ids=opts.gpu_id).cuda()
decoderRef_brdf = nn.DataParallel(network.RefineDecoderBRDF(), device_ids=opts.gpu_id).cuda()
decoderRef_render = nn.DataParallel(network.RefineDecoderRender(), device_ids=opts.gpu_id).cuda()

encoderRef2 = nn.DataParallel(network.RefineEncoder(), device_ids=opts.gpu_id).cuda()
decoderRef2_brdf = nn.DataParallel(network.RefineDecoderBRDF(), device_ids=opts.gpu_id).cuda()
decoderRef2_render = nn.DataParallel(network.RefineDecoderRender(), device_ids=opts.gpu_id).cuda()

render_layer = renderer.RenderLayerPointLightTorch()

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

path = '%s/models/pt' % (opts.outf)
encoder.load_state_dict(torch.load('%s/encoder.pth' % path, map_location=lambda storage, loc:storage))
decoder_brdf.load_state_dict(torch.load('%s/decoder_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoder_render.load_state_dict(torch.load('%s/decoder_render.pth' % path, map_location=lambda storage, loc:storage))

encoderRef.load_state_dict(torch.load('%s/encoderRef.pth' % path, map_location=lambda storage, loc:storage))
decoderRef_brdf.load_state_dict(torch.load('%s/decoderRef_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoderRef_render.load_state_dict(torch.load('%s/decoderRef_render.pth' % path, map_location=lambda storage, loc:storage))

encoderRef2.load_state_dict(torch.load('%s/encoderRef2.pth' % path, map_location=lambda storage, loc:storage))
decoderRef2_brdf.load_state_dict(torch.load('%s/decoderRef2_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoderRef2_render.load_state_dict(torch.load('%s/decoderRef2_render.pth' % path, map_location=lambda storage, loc:storage))


def gen_uniform_in_hemisphere():
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(0, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta) - 1
    return torch.from_numpy(np.array([[x, y, z]]))

def gen_light_batch_hemi(batch_size):
    light = torch.cat([gen_uniform_in_hemisphere() for i in range(batch_size)], dim=0)
    light = light.float().cuda()
    return light


dataset = evalset.SyntheticData(dataRoot=opts.data_root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, \
                    shuffle=opts.shuffle, pin_memory=True, num_workers=opts.workers)
opts.niter = len(dataloader)
prefetcher = evalset.DataPrefetcher(dataloader)
data = prefetcher.next()

ssim = SSIM()
msssim = MSSSIM()

mean_error_init = np.zeros((1, 8))
mean_error_cas  = np.zeros((1, 8))
mean_error_cas2 = np.zeros((1, 8))

name_list = []
err_list  = []

_i = 0
tic = time.time()
while data is not None:
    _i += 1

    albedo = data['albedo']
    normal = data['normal']
    rough  = data['rough']
    depth  = data['depth']
    seg    = data['seg']

    light_s = torch.zeros(1, 3).float().cuda()
    image_s = data['image_pt_src'] * seg

    light_t = data['light_pt_tar']
    image_t = data['image_pt_tar'] * seg

    input = torch.cat([image_s, seg], 1)
    feat = encoder(input)
    brdf_feat, brdf_pred = decoder_brdf(feat)
    albedo_pred, normal_pred, rough_pred, depth_pred = brdf_pred

    relit_render = 2 * render_layer.forward_batch(albedo_pred, normal_pred, rough_pred, depth_pred, seg, light_t) - 1
    
    relit_pred = decoder_render(feat, brdf_feat, light_t)

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

    relit_casrender = 2 * render_layer.forward_batch(albedo_pred_cas, normal_pred_cas, rough_pred_cas, depth_pred_cas, seg, light_t) - 1
    relit_caspred = decoderRef_render(feat_cas, brdf_feat_cas, light_t)

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

    relit_cas2render = 2 * render_layer.forward_batch(albedo_pred_cas2, normal_pred_cas2, rough_pred_cas2, depth_pred_cas2, seg, light_t) - 1
    relit_cas2pred = decoderRef2_render(feat_cas2, brdf_feat_cas2, light_t)

    error_init = torch.zeros(1, 8)
    pixel_num = (torch.sum(seg).cpu().data).item()
    error_init[0, 0] = torch.sum((albedo_pred - albedo) ** 2 * seg.expand_as(albedo)) / pixel_num / 3.0 / 4.0
    error_init[0, 1] = torch.sum((normal_pred - normal) ** 2 * seg.expand_as(normal)) / pixel_num / 3.0
    error_init[0, 2] = torch.sum((rough_pred  - rough) ** 2 * seg) / pixel_num / 4.0
    error_init[0, 3] = torch.sum((depth_pred  - depth) ** 2 * seg) / pixel_num
    error_init[0, 4] = torch.sum((relit_render - image_t) ** 2 * seg.expand_as(image_s)) / pixel_num / 3.0 / 4.0
    error_init[0, 5] = torch.sum((relit_pred - image_t) ** 2 * seg.expand_as(image_s)) / pixel_num / 3.0 / 4.0
    error_init[0, 6] = ssim((relit_pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_init[0, 7] = msssim((relit_pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)

    error_init[torch.isnan(error_init)] = 0
    mean_error_init += error_init.numpy()

    error_cas = torch.zeros(1, 8)
    pixel_num = (torch.sum(seg).cpu().data).item()
    error_cas[0, 0] = torch.sum((albedo_pred_cas - albedo) ** 2 * seg.expand_as(albedo)) / pixel_num / 3.0 / 4.0
    error_cas[0, 1] = torch.sum((normal_pred_cas - normal) ** 2 * seg.expand_as(normal)) / pixel_num / 3.0
    error_cas[0, 2] = torch.sum((rough_pred_cas  - rough) ** 2 * seg) / pixel_num / 4.0
    error_cas[0, 3] = torch.sum((depth_pred_cas  - depth) ** 2 * seg) / pixel_num
    error_cas[0, 4] = torch.sum((relit_casrender - image_t) ** 2 * seg.expand_as(image_s)) / pixel_num / 3.0 / 4.0
    error_cas[0, 5] = torch.sum((relit_caspred - image_t) ** 2 * seg.expand_as(image_s)) / pixel_num / 3.0 / 4.0
    error_cas[0, 6] = ssim((relit_caspred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_cas[0, 7] = msssim((relit_caspred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)

    error_cas[torch.isnan(error_cas)] = 0
    mean_error_cas += error_cas.numpy()

    error_cas2 = torch.zeros(1, 8)
    pixel_num = (torch.sum(seg).cpu().data).item()
    error_cas2[0, 0] = torch.sum((albedo_pred_cas2 - albedo) ** 2 * seg.expand_as(albedo)) / pixel_num / 3.0 / 4.0
    error_cas2[0, 1] = torch.sum((normal_pred_cas2 - normal) ** 2 * seg.expand_as(normal)) / pixel_num / 3.0
    error_cas2[0, 2] = torch.sum((rough_pred_cas2  - rough) ** 2 * seg) / pixel_num / 4.0
    error_cas2[0, 3] = torch.sum((depth_pred_cas2  - depth) ** 2 * seg) / pixel_num
    error_cas2[0, 4] = torch.sum((relit_cas2render - image_t) ** 2 * seg.expand_as(image_s)) / pixel_num / 3.0 / 4.0
    error_cas2[0, 5] = torch.sum((relit_cas2pred - image_t) ** 2 * seg.expand_as(image_s)) / pixel_num / 3.0 / 4.0
    error_cas2[0, 6] = ssim((relit_cas2pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_cas2[0, 7] = msssim((relit_cas2pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)

    error_cas2[torch.isnan(error_cas2)] = 0
    mean_error_cas2 += error_cas2.numpy()
    
    print('--> [%d / %d]' % (_i, opts.niter), mean_error_init/_i, mean_error_cas/_i, mean_error_cas2/_i)
    data = prefetcher.next()

mean_error_init = mean_error_init / opts.niter
mean_error_cas  = mean_error_cas  / opts.niter
mean_error_cas2 = mean_error_cas2 / opts.niter
print('--> final mean_error_init:', mean_error_init)
print('--> final mean_error_cas: ', mean_error_cas)
print('--> final mean_error_cas2:', mean_error_cas2)

toc = time.time()
print('--> total time used for evaluation: %f' % ((toc - tic) / 3600.0))
