import os
import json
import glob
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from time import time
from models import network, render
from torch.optim import lr_scheduler


class Model():
    """Build model"""
    def __init__(self, opts=None):
        """init"""
        self.opts  = opts
        self.name  = opts.name
        self.train = opts.train
        # build model
        self.build_net()
        # save options
        self.save_opt()
        SH = [[ 0.79,  0.44,  0.54],
              [ 0.39,  0.35,  0.60],
              [-0.34, -0.18, -0.27],
              [-0.29, -0.06,  0.01],
              [-0.11, -0.05, -0.12],
              [-0.26, -0.22, -0.47],
              [-0.16, -0.09, -0.15],
              [ 0.56,  0.21,  0.14],
              [ 0.21, -0.05, -0.30]]

        GCSH = torch.from_numpy(np.array(SH)).float().cuda()
        GCSH = GCSH.view(1, 3, 9).expand(opts.batch_size, 3, 9)
        # self.GCSH = GCSH

    def save_opt(self):
        """save training options to file"""
        path = '%s/%s' % (self.opts.outf, self.opts.name)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(vars(self.opts), f)

    def build_net(self):
        """Setup generator, optimizer, loss func and transfer to device
        """
        # Build net
        self.encoder = nn.DataParallel(network.encoderInitial(intc=7), device_ids=self.opts.gpu_id).cuda()
        self.decoder_brdf = nn.DataParallel(network.decoderBRDF(), device_ids=self.opts.gpu_id).cuda()
        self.decoder_render = nn.DataParallel(network.decoderRender(litc=30), device_ids=self.opts.gpu_id).cuda()
        # self.env_predictor = nn.DataParallel(network.envmapInitial(), device_ids=self.opts.gpu_id).cuda()

        self.render_layer = render.RenderLayerPointLightEnvTorch()

        print('--> loading saved model')
        path = '%s/%s/state_dict_15/models' % (self.opts.outf, self.name)
        self.encoder.load_state_dict(torch.load( '%s/encoder.pth' % path, map_location=lambda storage, loc:storage))
        self.decoder_brdf.load_state_dict(torch.load('%s/decoder_brdf.pth' % path, map_location=lambda storage, loc:storage))
        self.decoder_render.load_state_dict(torch.load('%s/decoder_render.pth' % path, map_location=lambda storage, loc:storage))
        # self.env_predictor.load_state_dict(torch.load('%s/env_predictor.pth' % path, map_location=lambda storage, loc:storage))
        
        def _fix(model):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        _fix(self.encoder)
        _fix(self.decoder_brdf)
        # _fix(self.env_predictor)
        
        # Optimizer
        self.w_brdf_A = 1
        self.w_brdf_N = 1
        self.w_brdf_R = 0.5
        self.w_brdf_D = 0.5
        self.w_env    = 0.01
        self.w_relit  = 1
        # Optimizer, actually only a group of optimizer
        self.optimizerDRen = torch.optim.Adam(self.decoder_render.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.error_list_relit  = []

        if self.opts.reuse:
            print('--> loading saved models and loss npys')
            [self.update_lr() for i in range(int(self.opts.start_epoch / 2))]
            self.load_saved_loss(self.opts.start_epoch)
            self.load_saved_checkpoint(self.opts.start_epoch)
        else:
            # loss for saving
            self.error_save_relit  = []
            print('--> start a new model')

    def set_input_var(self, data):
        """setup input var"""        
        self.albedo = data['albedo']
        self.normal = data['normal']
        self.rough  = data['rough']
        self.depth  = data['depth']
        self.mask   = data['seg']
        self.SH     = data['SH']
        self.image_bg = data['image_bg']

        self.light_s = torch.zeros(self.albedo.size(0), 3).float().cuda()
        # self.light_t = self.gen_light_batch_hemi(self.albedo.size(0))

        self.GCSH = torch.from_numpy(np.random.uniform(-1, 1, \
                    (self.albedo.size(0), 3, 9))).float().cuda()

        self.image_s_pe = self.make_image_under_pt_and_env(self.light_s, self.SH)
        self.image_t_pe = self.make_image_under_pt_and_env(self.light_s, self.GCSH)

    def gen_light_batch_hemi(self, batch_size):
        light = torch.cat([self.gen_uniform_in_hemisphere() for i in range(batch_size)], dim=0)
        light = light.float().cuda()
        return light

    def gen_uniform_in_hemisphere(self):
        """
        Randomly generate an unit 3D vector to represent
        the light direction
        """
        phi = np.random.uniform(0, np.pi * 2)
        costheta = np.random.uniform(0, 1)

        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta) - 1
        return torch.from_numpy(np.array([[x, y, z]]))

    def make_image_under_pt_and_env(self, light, SH):
        image_pt = self.render_layer.forward_batch(self.albedo, self.normal, self.rough, self.depth, self.mask, light) * self.mask
        image_env = self.render_layer.forward_env(self.albedo, self.normal, self.rough, self.mask, SH) + self.image_bg
        image_pe = 2 * (image_pt + image_env) - 1
        image_pe = torch.clamp_(image_pe, -1, 1)
        # image under point and env with bg
        return image_pe

    def forward(self):
        """forward: generate all data and GT"""
        input = torch.cat([self.image_s_pe, self.image_s_pe * self.mask, self.mask], 1)

        # encoder
        feat = self.encoder(input)
        # self.env_pred = self.env_predictor(feat[-1])

        brdf_feat, _ = self.decoder_brdf(feat)

        light_t = torch.cat([self.light_s, self.GCSH.view(self.albedo.size(0), -1)], 1) 
        self.relit_pred = self.decoder_render(feat, brdf_feat, light_t) * self.mask

    def compute_loss(self):
        pixel_num = (torch.sum(self.mask).cpu().data).item()
        self.loss_relit = torch.sum((self.relit_pred - self.image_t_pe) ** 2 * self.mask.expand_as(self.image_t_pe)) / pixel_num / 3.0
        self.loss = self.loss_relit

        self.error_list_relit.append(self.loss_relit.item())

    def _backward(self):
        self.compute_loss()
        self.loss.backward()

    def update(self):
        """update"""
        self.optimizerDRen.zero_grad()
        # forwarod
        self.forward()
        # update netG
        self._backward()
        self.optimizerDRen.step()

    def save_cur_sample(self, epoch):
        path = '%s/%s/env/state_dict_%s/samples' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        vutils.save_image(((((self.image_s_pe+1.0)/2.0))**(1.0/2.2)).data,
                    '{0}/image_src_bg.png'.format(path))

        vutils.save_image((((self.image_t_pe+1.0)/2.0 * self.mask + self.image_bg)**(1.0/2.2)).data,
                '{0}/image_gt.png'.format(path))
        vutils.save_image((((self.relit_pred+1.0)/2.0 * self.mask + self.image_bg)**(1.0/2.2)).data,
                '{0}/image_pred.png'.format(path))


    def flush_error_npy(self):
        self.error_save_relit.append(np.mean(self.error_list_relit))
        self.error_list_relit.clear()
        
    def save_error_to_file(self, epoch):
        path = '%s/%s/env/state_dict_%s/errors' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        np.save('{0}/relit_error_{1}.npy'.format(path, epoch), np.array(self.error_save_relit))

    def save_cur_checkpoint(self, epoch):
        print('--> saving checkpoints')
        path = '%s/%s/env/state_dict_%s/models' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.decoder_render.state_dict(), '%s/decoder_render.pth' % path)
        print('--> saving done')

    def load_saved_checkpoint(self, start_epoch):
        print('--> loading saved model')
        path = '%s/%s/env/state_dict_%s/models' % (self.opts.outf, self.name, str(start_epoch-1))
        self.decoder_render.load_state_dict(torch.load('%s/decoder_render.pth' % path, map_location=lambda storage, loc:storage))

    def load_saved_loss(self, epoch):
        epoch = epoch - 1
        path = '%s/%s/env/state_dict_%s/errors' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            raise ValueError('No such files: %s' % path)
        self.error_save_relit  = np.load('{0}/relit_error_{1}.npy'.format(path, epoch)).tolist()

    def update_lr(self, rate=2):
        print('--> devide lr by %d' % rate)
        for param_group in self.optimizerDRen.param_groups:
            param_group['lr'] /= rate

    def logger_loss(self, epoch, _iter):
        print('%s: [%d/%d][%d/%d], loss: %.4f' % (self.name, epoch, self.opts.nepoch[0], _iter, \
                        self.opts.niter, self.loss.item()))