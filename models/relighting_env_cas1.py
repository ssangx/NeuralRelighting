import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from models import network, renderer


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
        self.render_layer = renderer.RenderLayerPointLightEnvTorch()

        # Build net
        self.encoder = nn.DataParallel(network.encoderInitial(7), device_ids=self.opts.gpu_id).cuda()
        self.decoder_brdf = nn.DataParallel(network.decoderBRDF(), device_ids=self.opts.gpu_id).cuda()
        self.decoder_render = nn.DataParallel(network.decoderRender(litc=30), device_ids=self.opts.gpu_id).cuda()
        self.env_predictor = nn.DataParallel(network.envmapInitial(), device_ids=self.opts.gpu_id).cuda()

        print('--> loading saved model')
        path = '%s/%s/init/epoch_13/models' % (self.opts.outf, self.name)
        self.encoder.load_state_dict(torch.load( '%s/encoder.pth' % path, map_location=lambda storage, loc:storage))
        self.decoder_brdf.load_state_dict(torch.load('%s/decoder_brdf.pth' % path, map_location=lambda storage, loc:storage))
        self.decoder_render.load_state_dict(torch.load('%s/decoder_render.pth' % path, map_location=lambda storage, loc:storage))
        self.env_predictor.load_state_dict(torch.load('%s/env_predictor.pth' % path, map_location=lambda storage, loc:storage))

        def _fix(model):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        _fix(self.encoder)
        _fix(self.decoder_brdf)
        _fix(self.decoder_render)
        _fix(self.env_predictor)

        # Trainable
        self.encoderRef = nn.DataParallel(network.RefineEncoder(), device_ids=self.opts.gpu_id).cuda()
        self.decoderRef_brdf = nn.DataParallel(network.RefineDecoderBRDF(), device_ids=self.opts.gpu_id).cuda()
        self.decoderRef_render = nn.DataParallel(network.RefineDecoderRender(litc=30), device_ids=self.opts.gpu_id).cuda()
        self.env_caspredictor = nn.DataParallel(network.RefineDecoderEnv(), device_ids=self.opts.gpu_id).cuda()


        # Optimizer, actually only a group of optimizer
        self.optimizerE = torch.optim.Adam(self.encoderRef.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimizerBRDF = torch.optim.Adam(self.decoderRef_brdf.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizerDRen = torch.optim.Adam(self.decoderRef_render.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optimizerEnv = torch.optim.Adam(self.env_caspredictor.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.error_list_albedo = []
        self.error_list_normal = []
        self.error_list_depth  = []
        self.error_list_rough  = []
        self.error_list_env    = []
        self.error_list_relit  = []
        self.error_list_total  = []

        if self.opts.reuse:
            print('--> loading saved models and loss npys')
            [self.update_lr() for i in range(int(self.opts.start_epoch / 2))]
            self.load_saved_loss(self.opts.start_epoch)
            self.load_saved_checkpoint(self.opts.start_epoch)
        else:
            # loss for saving
            self.error_save_albedo = []
            self.error_save_normal = []
            self.error_save_depth  = []
            self.error_save_rough  = []
            self.error_save_env    = []
            self.error_save_relit  = []
            self.error_save_total  = []
            print('--> start a new model')

    def gen_light_batch(self, batch_size):
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
        self.image_s_pe = self.make_image_under_pt_and_env(self.light_s)

        self.light_t = self.gen_light_batch(self.albedo.size(0))
        self.image_t = self.make_image_under_pt_and_env(self.light_t)

    def make_image_under_pt_and_env(self, light):
        image_pt = self.render_layer.forward_batch(self.albedo, self.normal, self.rough, self.depth, self.mask, light) * self.mask
        image_env = self.render_layer.forward_env(self.albedo, self.normal, self.rough, self.mask, self.SH) + self.image_bg
        image_pe = 2 * (image_pt + image_env) - 1
        image_pe = torch.clamp_(image_pe, -1, 1)
        # image under point and env with bg
        return image_pe

    def forward(self):
        """forward: generate all data and GT"""
        input = torch.cat([self.image_s_pe, self.image_s_pe * self.mask, self.mask], 1)

        # encoder
        feat = self.encoder(input)

        self.env_pred = self.env_predictor(feat[-1])
        brdf_feat, brdf_pred = self.decoder_brdf(feat)
        self.albedo_pred, self.normal_pred, \
        self.rough_pred, self.depth_pred = brdf_pred

        light_t = torch.cat([self.light_t, self.env_pred.view(self.env_pred.size(0), -1)], 1) 
        self.relit_pred = self.decoder_render(feat, brdf_feat, light_t) * self.mask

        # render
        image_pt = self.render_layer.forward_batch(self.albedo_pred, self.normal_pred, self.rough_pred, self.depth_pred, self.mask, self.light_t)
        image_env = self.render_layer.forward_env(self.albedo_pred, self.normal_pred, self.rough_pred, self.mask, self.env_pred)
        relit_render = 2 * (image_pt + image_env) - 1
        self.relit_render = torch.clamp_(relit_render, -1, 1) * self.mask

        # ----------------- Cas1 -----------------
        cas_input = torch.cat([self.image_s_pe, \
                               self.relit_pred * self.mask, \
                               self.relit_render * self.mask, \
                               self.albedo_pred * self.mask, \
                               self.normal_pred * self.mask, \
                               self.rough_pred * self.mask, \
                               self.depth_pred * self.mask, \
                               self.mask, \
                               ], dim=1)

        feat_cas = self.encoderRef(cas_input)
        brdf_feat_cas, brdf_pred_cas = self.decoderRef_brdf(feat_cas)
        self.albedo_pred_cas, self.normal_pred_cas, \
        self.rough_pred_cas, self.depth_pred_cas = brdf_pred_cas

        self.env_pred_cas = self.env_caspredictor(feat_cas[-1], self.env_pred)
        light_t = torch.cat([self.light_t, self.env_pred_cas.view(self.env_pred_cas.size(0), -1)], 1) 
        self.relit_caspred = self.decoderRef_render(feat_cas, brdf_feat_cas, light_t) * self.mask

    def compute_loss(self):
        pixel_num = (torch.sum(self.mask).cpu().data).item()

        self.loss_relit = torch.sum((self.relit_caspred - self.image_t) ** 2 * self.mask.expand_as(self.image_t)) / pixel_num / 3.0

        self.lossA = torch.sum((self.albedo_pred_cas - self.albedo) ** 2 * self.mask.expand_as(self.albedo)) / pixel_num / 3.0
        self.lossN = torch.sum((self.normal_pred_cas - self.normal) ** 2 * self.mask.expand_as(self.normal)) / pixel_num / 3.0
        self.lossR = torch.sum((self.rough_pred_cas  - self.rough)  ** 2 * self.mask) / pixel_num
        self.lossD = torch.sum((self.depth_pred_cas  - self.depth)  ** 2 * self.mask) / pixel_num

        self.lossE = torch.mean((self.SH - self.env_pred_cas) ** 2)

        self.loss = 1 * self.lossA + \
                    1 * self.lossN + \
                    0.5 * self.lossR + \
                    0.5  * self.lossD + \
                    1 * self.loss_relit + \
                    0.01 * self.lossE

        self.error_list_albedo.append(self.lossA.item())
        self.error_list_normal.append(self.lossN.item())
        self.error_list_depth.append(self.lossD.item())
        self.error_list_rough.append(self.lossR.item())
        self.error_list_env.append(self.lossE.item())
        self.error_list_relit.append(self.loss_relit.item())
        self.error_list_total.append(self.loss.item())

    def _backward(self):
        self.compute_loss()
        self.loss.backward()

    def update(self):
        """update"""
        self.optimizerE.zero_grad()
        self.optimizerBRDF.zero_grad()
        self.optimizerDRen.zero_grad()
        self.optimizerEnv.zero_grad()
        # forwarod
        self.forward()
        # backward
        self._backward()
        self.optimizerE.step()
        self.optimizerBRDF.step()
        self.optimizerDRen.step()
        self.optimizerEnv.step()

    def save_cur_sample(self, epoch):
        path = '%s/%s/cas1/epoch_%s/samples' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        vutils.save_image(((((self.image_s_pe+1.0)/2.0))**(1.0/2.2)).data,
                    '{0}/image_src_bg.png'.format(path))        
        vutils.save_image(((self.image_bg)**(1.0/2.2)).data,
                    '{0}/image_bg.png'.format(path))
                    
        vutils.save_image((((self.relit_pred+1.0)/2.0 + self.image_bg)**(1.0/2.2)).data,
                    '{0}/relit_pred.png'.format(path))
        vutils.save_image((((self.relit_caspred+1.0)/2.0 + self.image_bg)**(1.0/2.2)).data,
                    '{0}/relit_caspred.png'.format(path))

        vutils.save_image((0.5*(self.albedo + 1)*self.mask.expand_as(self.albedo)).data,
                    '{0}/albedo_gt.png'.format(path))
        vutils.save_image((0.5*(self.normal + 1)*self.mask.expand_as(self.normal)).data,
                    '{0}/normal_gt.png'.format(path))
        vutils.save_image((0.5*(self.rough  + 1)*self.mask.expand_as(self.rough )).data,
                    '{0}/rough_gt.png'.format(path))
        depth = 1 / torch.clamp(self.depth, 1e-6, 10) * self.mask.expand_as(self.depth)
        depth = (depth - 0.25) / 0.8
        vutils.save_image((depth*self.mask.expand_as(depth)).data,'{0}/depth_gt.png'.format(path))

        vutils.save_image((0.5*(self.albedo_pred_cas + 1)*self.mask.expand_as(self.albedo)).data,
                    '{0}/albedo_pred.png'.format(path))
        vutils.save_image((0.5*(self.normal_pred_cas + 1)*self.mask.expand_as(self.normal)).data,
                    '{0}/normal_pred.png'.format(path))
        vutils.save_image((0.5*(self.rough_pred_cas  + 1)*self.mask.expand_as(self.rough )).data,
                    '{0}/rough_pred.png'.format(path))
        depth = 1 / torch.clamp(self.depth_pred_cas, 1e-6, 10) * self.mask.expand_as(self.depth)
        depth = (depth - 0.25) / 0.8
        vutils.save_image((depth*self.mask.expand_as(depth)).data,'{0}/depth_pred.png'.format(path))

    def flush_error_npy(self):
        self.error_save_albedo.append(np.mean(self.error_list_albedo))
        self.error_save_normal.append(np.mean(self.error_list_normal))
        self.error_save_depth.append(np.mean(self.error_list_depth))
        self.error_save_rough.append(np.mean(self.error_list_rough))
        self.error_save_relit.append(np.mean(self.error_list_relit))
        self.error_save_total.append(np.mean(self.error_list_total))
        self.error_save_env.append(np.mean(self.error_list_env))

        self.error_list_albedo.clear()
        self.error_list_normal.clear()
        self.error_list_depth.clear()
        self.error_list_rough.clear()
        self.error_list_relit.clear()
        self.error_list_total.clear()
        self.error_list_env.clear()
        
    def save_error_to_file(self, epoch):
        path = '%s/%s/cas1/epoch_%s/errors' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        np.save('{0}/albedo_error_{1}.npy'.format(path, epoch), np.array(self.error_save_albedo))
        np.save('{0}/normal_error_{1}.npy'.format(path, epoch), np.array(self.error_save_normal))
        np.save('{0}/rough_error_{1}.npy'.format(path, epoch), np.array(self.error_save_rough))
        np.save('{0}/depth_error_{1}.npy'.format(path, epoch), np.array(self.error_save_depth))
        np.save('{0}/relit_error_{1}.npy'.format(path, epoch), np.array(self.error_save_relit))      
        np.save('{0}/total_error_{1}.npy'.format(path, epoch), np.array(self.error_save_total))   
        np.save('{0}/env_error_{1}.npy'.format(path, epoch), np.array(self.error_save_env))

    def save_cur_checkpoint(self, epoch):
        print('--> saving checkpoints')
        path = '%s/%s/cas1/epoch_%s/models' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.encoderRef.state_dict(),  '%s/encoderRef.pth'  % path)
        torch.save(self.decoderRef_brdf.state_dict(), '%s/decoderRef_brdf.pth' % path)
        torch.save(self.decoderRef_render.state_dict(), '%s/decoderRef_render.pth' % path)
        torch.save(self.env_caspredictor.state_dict(), '%s/env_caspredictor.pth' % path)
        print('--> saving done')

    def load_saved_checkpoint(self, start_epoch):
        print('--> loading saved model')
        path = '%s/%s/cas1/epoch_%s/models' % (self.opts.outf, self.name, str(start_epoch-1))
        self.encoderRef.load_state_dict(torch.load( '%s/encoderRef.pth'  % path, map_location=lambda storage, loc:storage))
        self.decoderRef_brdf.load_state_dict(torch.load('%s/decoderRef_brdf.pth' % path, map_location=lambda storage, loc:storage))
        self.decoderRef_render.load_state_dict(torch.load('%s/decoderRef_render.pth' % path, map_location=lambda storage, loc:storage))
        self.env_caspredictor.load_state_dict(torch.load('%s/env_caspredictor.pth' % path, map_location=lambda storage, loc:storage))

    def load_saved_loss(self, epoch):
        epoch = epoch - 1
        path = '%s/%s/cas1/epoch_%s/errors' % (self.opts.outf, self.name, str(epoch))
        if not os.path.exists(path):
            raise ValueError('No such files: %s' % path)
        self.error_save_albedo = np.load('{0}/albedo_error_{1}.npy'.format(path, epoch)).tolist()
        self.error_save_normal = np.load('{0}/normal_error_{1}.npy'.format(path, epoch)).tolist()
        self.error_save_rough  = np.load('{0}/rough_error_{1}.npy'.format(path, epoch)).tolist()
        self.error_save_depth  = np.load('{0}/depth_error_{1}.npy'.format(path, epoch)).tolist()
        self.error_save_relit  = np.load('{0}/relit_error_{1}.npy'.format(path, epoch)).tolist()
        self.error_save_total  = np.load('{0}/total_error_{1}.npy'.format(path, epoch)).tolist()
        self.error_save_env    = np.load('{0}/env_error_{1}.npy'.format(path, epoch)).tolist()

    def update_lr(self, rate=2):
        print('--> devide lr by %d' % rate)
        for param_group in self.optimizerE.param_groups:
            param_group['lr'] /= rate
        for param_group in self.optimizerBRDF.param_groups:
            param_group['lr'] /= rate
        for param_group in self.optimizerDRen.param_groups:
            param_group['lr'] /= rate
        for param_group in self.optimizerEnv.param_groups:
            param_group['lr'] /= rate

    def print_loss(self, epoch, _iter):
        print('%s: [%d/%d][%d/%d], loss: %.4f' % (self.name, epoch, self.opts.nepoch[0], _iter, \
                        self.opts.niter, self.loss.item()))
        print('A: %.3f, N: %.3f, R: %.3f, D: %.3f, relit: %.3f, env: %.3f' % (self.lossA.item(), self.lossN.item(), \
                     self.lossR.item(), self.lossD.item(), self.loss_relit.item(), self.lossE.item()))