#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:33:04 2018

@author: chinwei
"""

import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
#import torch.optim as optim
from torchkit import optim
from torch.autograd import Variable
from torchkit import nn as nn_, flows, utils
from torchkit.transforms import from_numpy, binarize
from torchvision.transforms import transforms
from ops import load_bmnist_image, load_omniglot_image, load_mnist_image
from ops import DatasetWrapper
from itertools import chain



import time 
import json
import argparse, os


class VAE(object):
    
    def __init__(self, args):

        self.args = args        
        self.__dict__.update(args.__dict__)
        
        dimz = args.dimz
        dimc = args.dimc
        dimh = args.dimh
        flowtype = args.flowtype
        num_flow_layers = args.num_flow_layers
        num_ds_dim = args.num_ds_dim
        num_ds_layers = args.num_ds_layers
                 
                 
        act = nn.ELU()
        if flowtype == 'affine':
            flow = flows.IAF
        elif flowtype == 'dsf':
            flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                                 num_ds_layers=num_ds_layers,
                                                 **kwargs)
        elif flowtype == 'ddsf':
            flow = lambda **kwargs:flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                                  num_ds_layers=num_ds_layers,
                                                  **kwargs)
        
        self.enc = nn.Sequential(
                nn_.ResConv2d(1,16,3,2,padding=1,activation=act),
                act,
                nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
                act,
                nn_.ResConv2d(16,32,3,2,padding=1,activation=act),
                act,
                nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
                act,
                nn_.ResConv2d(32,32,3,2,padding=1,activation=act),
                act,
                nn_.Reshape((-1,32*4*4)),
                nn_.ResLinear(32*4*4,dimc),
                act
                )
        
        self.inf = nn.Sequential( 
                flows.LinearFlow(dimz, dimc),
                *[nn_.SequentialFlow(
                    flow(dim=dimz,
                         hid_dim=dimh,
                         context_dim=dimc,
                         num_layers=2,
                         activation=act),
                    flows.FlipFlow(1)) for i in range(num_flow_layers)])
        
        self.dec = nn.Sequential(
                nn_.ResLinear(dimz,dimc),
                act,
                nn_.ResLinear(dimc,32*4*4),
                act,
                nn_.Reshape((-1,32,4,4)),
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
                act,
                nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
                act,
                nn_.slicer[:,:,:-1,:-1],                
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn_.ResConv2d(32,16,3,1,padding=1,activation=act),
                act,
                nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
                act,
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn_.ResConv2d(16,1,3,1,padding=1,activation=act),
                )

        self.dec[-1].conv_01.bias.data.normal_(-3, 0.0001)

        if self.cuda:
            self.enc = self.enc.cuda()
            self.inf = self.inf.cuda()
            self.dec = self.dec.cuda()            
        
        
        amsgrad = bool(args.amsgrad)
        polyak = args.polyak
        self.optim = optim.Adam(chain(self.enc.parameters(),
                                      self.inf.parameters(),
                                      self.dec.parameters()),
                                lr=args.lr, 
                                betas=(args.beta1, args.beta2),
                                amsgrad=amsgrad,
                                polyak=polyak)
        
        
    
    def loss(self, x, weight=1.0, bits=0.0):
        n = x.size(0)
        zero = utils.varify(np.zeros(1).astype('float32'))
        context = self.enc(x)
        
        ep = utils.varify(np.random.randn(n,self.dimz).astype('float32'))
        lgd = utils.varify(np.zeros(n).astype('float32'))
        if self.cuda:
            ep = ep.cuda()
            lgd = lgd.cuda()
            zero = zero.cuda()
            
        z, logdet, _ = self.inf((ep, lgd, context))
        pi = nn_.sigmoid(self.dec(z))
        
        logpx = - utils.bceloss(pi, x).sum(1).sum(1).sum(1)
        logqz = utils.log_normal(ep, zero, zero).sum(1) - logdet
        logpz = utils.log_normal(z, zero, zero).sum(1)
        kl = logqz - logpz
        
        return - (logpx - torch.max(kl*weight, torch.ones_like(kl)*bits)), \
               - logpx, kl
        
    def iwlb(self, x, niw=1):
        LOSSES = list()
        for i in range(niw):
            LOSSES.append(sum(self.loss(x,1.0)[1:])[:,None].data.cpu().numpy())
        return -utils.log_mean_exp_np(-np.concatenate(LOSSES, 1))
        
    def state_dict(self):
        return self.enc.state_dict(), \
               self.inf.state_dict(), \
               self.dec.state_dict()

    def load_state_dict(self, states):
        self.enc.load_state_dict(states[0]) 
        self.inf.load_state_dict(states[1])
        self.dec.load_state_dict(states[2])       

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(chain(self.enc.parameters(),
                                      self.inf.parameters(),
                                      self.dec.parameters()),
                                self.clip)



class model(object):
    
    def __init__(self, args, filename):
        
        self.__dict__.update(args.__dict__)

        self.filename = filename
        self.args = args        
        if args.dataset == 'sb_mnist':
            tr, va, te = load_bmnist_image()
        elif args.dataset == 'db_mnist':
            tr, va, te = load_mnist_image()
        elif args.dataset == 'db_omniglot':
            tr, va, te = load_omniglot_image()
        else:
            raise Exception('dataset {} not supported'.format(args.dataset))
        
        if args.final_mode:
            tr = np.concatenate([tr, va], axis=0)
            va = te[:]
        
        if args.dataset[:2] == 'db':
            compose = transforms.Compose([from_numpy(), binarize()])
            tr = DatasetWrapper(tr,compose)
            va = DatasetWrapper(va,compose)
            te = DatasetWrapper(te,compose)
            
        self.train_loader = data.DataLoader(tr, 
                                            batch_size=args.batch_size,
                                            shuffle=True)
        self.valid_loader = data.DataLoader(va, 
                                            batch_size=args.batch_size,
                                            shuffle=False)
        self.test_loader = data.DataLoader(te, 
                                           batch_size=args.batch_size,
                                           shuffle=False)
        
        
        self.vae = VAE(args)
        
        
    def train(self, epoch, total=10000):
        optim = self.vae.optim
        t = 0 
        
        LOSSES = 0
        KLS = 0
        RECS = 0
        counter = 0
        best_val = float('inf')
        for e in range(epoch):            
            for x in self.train_loader:
                optim.zero_grad()
                weight = min(1.0,max(self.anneal0, t/float(total)))
                x = Variable(x).view(-1,1,28,28)
                if self.cuda:
                    x = x.cuda()
                    
                losses_, rec, kl = self.vae.loss(x, weight, self.bits)
                losses_.mean().backward()
                losses = rec + kl
                LOSSES += losses.sum().data.cpu().numpy()
                RECS += rec.sum().data.cpu().numpy()
                KLS += kl.sum().data.cpu().numpy()
                counter += losses.size(0)

                self.vae.clip_grad_norm()
                optim.step()
                t += 1

            if e%1 == 0:      
                self.vae.optim.swap()
                loss_val = self.evaluate(self.valid_loader, 1)
                loss_tst = self.evaluate(self.test_loader, 1)       
                print 'Epoch: [%4d/%4d] train <= %.2f %.2f %.2f ' \
                      'valid: %.3f test: %.3f' % \
                (e+1, epoch, LOSSES/float(counter), 
                 RECS/float(counter), KLS/float(counter),
                 loss_val,
                 loss_tst), weight
                if loss_val < best_val:
                    print(' [^] Best validation loss [^] ... [saving]')
                    self.save(self.save_dir+'/'+self.filename+'_best')
                    best_val = loss_val
                LOSSES = 0
                RECS = 0
                KLS = 0
                counter = 0
                self.vae.optim.swap()
            if (e+1)%5 == 0:
                self.save(self.save_dir+'/'+self.filename+'_last')

        
        if not self.final_mode:
            # loading best valid model (early stopping)
            self.load(self.save_dir+'/'+self.filename+'_best')


    def evaluate(self, dataloader, niw=1):
        LOSSES = 0 
        c = 0
        for x in dataloader:
            x = Variable(x).view(-1,1,28,28)
            if self.cuda:
                x = x.cuda()
                
            losses = self.vae.iwlb(x, niw)
            LOSSES += losses.sum()
            c += losses.shape[0]
        return LOSSES / float(c)


    def save(self, fn):        
        torch.save(self.vae.state_dict(), fn+'_model.pt')
        with open(fn+'_args.txt','w') as out:
            out.write(json.dumps(self.args.__dict__,indent=4))

    def load(self, fn):
        self.vae.load_state_dict(torch.load(fn+'_model.pt'))
            

# =============================================================================
# main
# =============================================================================


"""parsing and configuration"""
def parse_args():
    desc = "VAE"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='db_omniglot', 
                        choices=['sb_mnist', 
                                 'db_mnist',
                                 'db_omniglot'],
                        help='static/dynamic binarized mnist')
    parser.add_argument('--epoch', type=int, default=400, 
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--seed', type=int, default=1993,
                        help='Random seed')
    parser.add_argument('--fn', type=str, default='0',
                        help='Filename of model to be loaded')
    parser.add_argument('--to_train', type=int, default=1,
                        help='1 if to train 0 if not')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--anneal', type=int, default=50000)
    parser.add_argument('--anneal0', type=float, default=0.0001)
    parser.add_argument('--bits', type=float, default=0.10)
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--polyak', type=float, default=0.998)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--final_mode', default=False, action='store_true')
    


    parser.add_argument('--dimz', type=int, default=32)
    parser.add_argument('--dimc', type=int, default=450)
    parser.add_argument('--dimh', type=int, default=1920)
    parser.add_argument('--flowtype', type=str, default='affine')
    parser.add_argument('--num_flow_layers', type=int, default=0)
    parser.add_argument('--num_ds_dim', type=int, default=16)
    parser.add_argument('--num_ds_layers', type=int, default=1)
    
    
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir + '_' + args.dataset):
        os.makedirs(args.result_dir + '_' + args.dataset)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args
   

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed+10000)

    fn = str(time.time()).replace('.','')
    print args
    print fn

    print(" [*] Building model!")    
    old_fn = args.save_dir+'/'+args.fn+'_args.txt'
    if os.path.isfile(old_fn):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        d = without_keys(json.loads(open(old_fn,'r').read()),
                         ['to_train','epoch','anneal'])
        args.__dict__.update(d)
        print(" New args:" )
        print args
        mdl = model(args, fn)
        print(" [*] Loading model!")
        mdl.load(args.save_dir+'/'+args.fn)
    else:
        mdl = model(args, fn)
    
    # launch the graph in a session
    if args.to_train:
        print(" [*] Training started!")
        mdl.train(args.epoch, args.anneal)
        print(" [*] Training finished!")

    print " [**] Valid: %.4f" % mdl.evaluate(mdl.valid_loader, 2000)
    print " [**] Test: %.4f" % mdl.evaluate(mdl.test_loader, 2000)

    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()

    
    
