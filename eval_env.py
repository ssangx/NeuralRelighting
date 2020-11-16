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
encoder = nn.DataParallel(network.encoderInitial(7), device_ids=opts.gpu_id).cuda()
decoder_brdf = nn.DataParallel(network.decoderBRDF(), device_ids=opts.gpu_id).cuda()
decoder_render = nn.DataParallel(network.decoderRender(litc=30), device_ids=opts.gpu_id).cuda()
env_predictor = nn.DataParallel(network.envmapInitial(), device_ids=opts.gpu_id).cuda()

encoderRef = nn.DataParallel(network.RefineEncoder(), device_ids=opts.gpu_id).cuda()
decoderRef_brdf = nn.DataParallel(network.RefineDecoderBRDF(), device_ids=opts.gpu_id).cuda()
decoderRef_render = nn.DataParallel(network.RefineDecoderRender(litc=30), device_ids=opts.gpu_id).cuda()
env_caspredictor = nn.DataParallel(network.RefineDecoderEnv(), device_ids=opts.gpu_id).cuda()

encoderRef2 = nn.DataParallel(network.RefineEncoder(), device_ids=opts.gpu_id).cuda()
decoderRef2_brdf = nn.DataParallel(network.RefineDecoderBRDF(), device_ids=opts.gpu_id).cuda()
decoderRef2_render = nn.DataParallel(network.RefineDecoderRender(litc=30), device_ids=opts.gpu_id).cuda()
env_cas2predictor = nn.DataParallel(network.RefineDecoderEnv(), device_ids=opts.gpu_id).cuda()

render_layer = renderer.RenderLayerPointLightEnvTorch()

path = '%s/models/env' % (opts.outf)
encoder.load_state_dict(torch.load('%s/encoder.pth' % path, map_location=lambda storage, loc:storage))
decoder_brdf.load_state_dict(torch.load('%s/decoder_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoder_render.load_state_dict(torch.load('%s/decoder_render.pth' % path, map_location=lambda storage, loc:storage))

path = '%s/models/env' % (opts.outf)
encoderRef.load_state_dict(torch.load('%s/encoderRef.pth' % path, map_location=lambda storage, loc:storage))
decoderRef_brdf.load_state_dict(torch.load('%s/decoderRef_brdf.pth' % path, map_location=lambda storage, loc:storage))
decoderRef_render.load_state_dict(torch.load('%s/decoderRef_render.pth' % path, map_location=lambda storage, loc:storage))

path = '%s/models/env' % (opts.outf)
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
_fix(env_predictor)

_fix(encoderRef)
_fix(decoderRef_brdf)
_fix(decoderRef_render)
_fix(env_caspredictor)

_fix(encoderRef2)
_fix(decoderRef2_brdf)
_fix(decoderRef2_render)
_fix(env_cas2predictor)

dataset = evalset.SyntheticData(dataRoot=opts.data_root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, \
                    shuffle=opts.shuffle, pin_memory=True, num_workers=opts.workers)
opts.niter = len(dataloader)
prefetcher = evalset.DataPrefetcher(dataloader)
data = prefetcher.next()

ssim = SSIM()
msssim = MSSSIM()

mean_error_init = np.zeros((1, 9))
mean_error_cas  = np.zeros((1, 9))
mean_error_cas2 = np.zeros((1, 9))

def make_image_under_pt_and_env(A, N, R, D, S, light, env, bg):
    image_pt = render_layer.forward_batch(A, N, R, D, S, light) * S
    image_env = render_layer.forward_env(A, N, R, S, env) + bg
    image_pe = 2 * (image_pt + image_env) - 1
    image_pe = torch.clamp_(image_pe, -1, 1)
    return image_pe

_i = 0
tic = time.time()
while data is not None:
    _i += 1

    albedo = data['albedo']
    normal = data['normal']
    rough  = data['rough']
    depth  = data['depth']
    seg    = data['seg']
    SH     = data['SH']

    image_bg = data['image_bg']
    image_pe = data['image_env_src']
    light_s = torch.zeros(1, 3).float().cuda()

    image_t = data['image_env_tar']
    light_t = data['light_env_tar']
    
    input = torch.cat([image_pe, image_pe * seg, seg], 1)

    feat = encoder(input)

    env_pred = env_predictor(feat[-1])

    brdf_feat, brdf_pred = decoder_brdf(feat)
    albedo_pred, normal_pred, rough_pred, depth_pred = brdf_pred

    light_t_env = torch.cat([light_t, env_pred.view(input.size(0), -1)], 1) 
    relit_pred = decoder_render(feat, brdf_feat, light_t_env) * seg

    relit_render = make_image_under_pt_and_env(albedo_pred, normal_pred, rough_pred, depth_pred, seg, light_t, env_pred, image_bg) * seg

    # Cas1
    cas_input = torch.cat([image_pe, \
                           relit_pred  * seg, \
                           relit_render * seg, \
                           albedo_pred * seg, \
                           normal_pred * seg, \
                           rough_pred  * seg, \
                           depth_pred  * seg, \
                           seg, \
                           ], dim=1)

    feat_cas = encoderRef(cas_input)
    brdf_feat_cas, brdf_pred_cas = decoderRef_brdf(feat_cas)
    albedo_pred_cas, normal_pred_cas, \
    rough_pred_cas, depth_pred_cas = brdf_pred_cas

    env_pred_cas = env_caspredictor(feat_cas[-1], env_pred)
    light_t_env = torch.cat([light_t, env_pred_cas.view(env_pred_cas.size(0), -1)], 1) 
    relit_caspred = decoderRef_render(feat_cas, brdf_feat_cas, light_t_env) * seg

    relit_casrender = make_image_under_pt_and_env(albedo_pred_cas, normal_pred_cas, rough_pred_cas, depth_pred_cas, seg, light_t, env_pred_cas, image_bg) * seg

    # Cas2
    cas2_input = torch.cat([image_pe, \
                           relit_caspred  * seg, \
                           relit_casrender * seg, \
                           albedo_pred_cas * seg, \
                           normal_pred_cas * seg, \
                           rough_pred_cas  * seg, \
                           depth_pred_cas  * seg, \
                           seg, \
                           ], dim=1)

    feat_cas2 = encoderRef2(cas2_input)
    brdf_feat_cas2, brdf_pred_cas2 = decoderRef2_brdf(feat_cas2)
    albedo_pred_cas2, normal_pred_cas2, \
    rough_pred_cas2, depth_pred_cas2 = brdf_pred_cas2

    env_pred_cas2 = env_cas2predictor(feat_cas2[-1], env_pred_cas)
    light_t_env = torch.cat([light_t, env_pred_cas2.view(env_pred_cas2.size(0), -1)], 1) 
    relit_cas2pred = decoderRef2_render(feat_cas2, brdf_feat_cas2, light_t_env) * seg

    relit_cas2render = make_image_under_pt_and_env(albedo_pred_cas2, normal_pred_cas2, rough_pred_cas2, depth_pred_cas2, seg, light_t, env_pred_cas2, image_bg) * seg

    # Eval
    pixel_num = (torch.sum(seg).cpu().data).item()

    error_init = torch.zeros(1, 9)
    error_init[0, 0] = torch.sum((albedo_pred - albedo) ** 2 * seg.expand_as(albedo)) / pixel_num / 3.0 / 4.0
    error_init[0, 1] = torch.sum((normal_pred - normal) ** 2 * seg.expand_as(normal)) / pixel_num / 3.0
    error_init[0, 2] = torch.sum((rough_pred  - rough) ** 2 * seg) / pixel_num / 4.0
    error_init[0, 3] = torch.sum((depth_pred  - depth) ** 2 * seg) / pixel_num
    error_init[0, 4] = torch.mean((env_pred  - SH) ** 2)
    error_init[0, 5] = torch.sum((relit_render - image_t) ** 2 * seg.expand_as(image_pe)) / pixel_num / 3.0 / 4.0
    error_init[0, 6] = torch.sum((relit_pred - image_t) ** 2 * seg.expand_as(image_pe)) / pixel_num / 3.0 / 4.0
    error_init[0, 7] = ssim((relit_pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_init[0, 8] = msssim((relit_pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_init[torch.isnan(error_init)] = 0
    mean_error_init += error_init.numpy()

    error_cas = torch.zeros(1, 9)
    error_cas[0, 0] = torch.sum((albedo_pred_cas - albedo) ** 2 * seg.expand_as(albedo)) / pixel_num / 3.0 / 4.0
    error_cas[0, 1] = torch.sum((normal_pred_cas - normal) ** 2 * seg.expand_as(normal)) / pixel_num / 3.0
    error_cas[0, 2] = torch.sum((rough_pred_cas  - rough) ** 2 * seg) / pixel_num / 4.0
    error_cas[0, 3] = torch.sum((depth_pred_cas  - depth) ** 2 * seg) / pixel_num
    error_cas[0, 4] = torch.mean((env_pred_cas  - SH) ** 2)
    error_cas[0, 5] = torch.sum((relit_casrender - image_t) ** 2 * seg.expand_as(image_pe)) / pixel_num / 3.0 / 4.0
    error_cas[0, 6] = torch.sum((relit_caspred - image_t) ** 2 * seg.expand_as(image_pe)) / pixel_num / 3.0 / 4.0
    error_cas[0, 7] = ssim((relit_caspred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_cas[0, 8] = msssim((relit_caspred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_cas[torch.isnan(error_cas)] = 0
    mean_error_cas += error_cas.numpy()

    error_cas2 = torch.zeros(1, 9)
    error_cas2[0, 0] = torch.sum((albedo_pred_cas2 - albedo) ** 2 * seg.expand_as(albedo)) / pixel_num / 3.0 / 4.0
    error_cas2[0, 1] = torch.sum((normal_pred_cas2 - normal) ** 2 * seg.expand_as(normal)) / pixel_num / 3.0
    error_cas2[0, 2] = torch.sum((rough_pred_cas2  - rough) ** 2 * seg) / pixel_num / 4.0
    error_cas2[0, 3] = torch.sum((depth_pred_cas2  - depth) ** 2 * seg) / pixel_num
    error_cas2[0, 4] = torch.mean((env_pred_cas2  - SH) ** 2)
    error_cas2[0, 5] = torch.sum((relit_cas2render - image_t) ** 2 * seg.expand_as(image_pe)) / pixel_num / 3.0 / 4.0
    error_cas2[0, 6] = torch.sum((relit_cas2pred - image_t) ** 2 * seg.expand_as(image_pe)) / pixel_num / 3.0 / 4.0
    error_cas2[0, 7] = ssim((relit_cas2pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_cas2[0, 8] = msssim((relit_cas2pred + 1) * 0.5 * seg, (image_t + 1) * 0.5 * seg)
    error_cas2[torch.isnan(error_cas2)] = 0
    mean_error_cas2 += error_cas2.numpy()
    
    print('--> [%d / %d]' % (_i, opts.niter), mean_error_init / _i, mean_error_cas / _i)
    data = prefetcher.next()

mean_error_init = mean_error_init / opts.niter
mean_error_cas  = mean_error_cas / opts.niter
mean_error_cas2 = mean_error_cas2 / opts.niter
print('--> final mean_error_init:', mean_error_init)
print('--> final mean_error_cas: ', mean_error_cas)
print('--> final mean_error_cas2:', mean_error_cas2)

toc = time.time()
print('--> total time used for evaluation: %f' % ((toc - tic) / 3600.0))
