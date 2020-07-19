import os
import torch
import numpy as np
import torchvision.utils as vutils


def print_options(opts):
    strs = '------------ Options -------------\n'
    strs += '{}'.format(dictToString(vars(opts)))
    strs += '-------------- End ----------------\n'
    print(strs)


def dictToString(dicts, start='', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs


def print_training(opts, model, epoch, iter):
    if opts.netG == 'unet64':
        print('[%d/%d][%d/%d] Loss: %.4f'
                % (epoch, opts.nepoch, iter, opts.niter, model.loss_G))
    else:
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                % (epoch, opts.nepoch, iter, opts.niter, model.cur_loss_D, model.cur_loss_G))