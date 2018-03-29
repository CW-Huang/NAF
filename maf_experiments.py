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
from ops import load_maf_data
from ops import DatasetWrapper




import time 
import json
import argparse, os


class MAF(object):
    
    def __init__(self, args, p):

        self.args = args        
        self.__dict__.update(args.__dict__)
        self.p = p
        
        dim = p
        dimc = 1
        dimh = args.dimh
        flowtype = args.flowtype
        num_flow_layers = args.num_flow_layers
        num_ds_dim = args.num_ds_dim
        num_ds_layers = args.num_ds_layers
        fixed_order = args.fixed_order
                 
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
        
        
        sequels = [nn_.SequentialFlow(
            flow(dim=dim,
                 hid_dim=dimh,
                 context_dim=dimc,
                 num_layers=args.num_hid_layers+1,
                 activation=act,
                 fixed_order=fixed_order),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + \
                  [flows.LinearFlow(dim, dimc),]
                
                
        self.flow = nn.Sequential(
                *sequels)
        
        
        
        if self.cuda:
            self.flow = self.flow.cuda()
        
        
        
        
    def density(self, spl):
        n = spl.size(0)
        context = Variable(torch.FloatTensor(n, 1).zero_()) 
        lgd = Variable(torch.FloatTensor(n).zero_())
        zeros = Variable(torch.FloatTensor(n, self.p).zero_())
        if self.cuda:
            context = context.cuda()
            lgd = lgd.cuda()
            zeros = zeros.cuda()
            
        z, logdet, _ = self.flow((spl, lgd, context))
        losses = - utils.log_normal(z, zeros, zeros+1.0).sum(1) - logdet
        return - losses

    def loss(self, x):
        return - self.density(x)
        
    def state_dict(self):
        return self.flow.state_dict()

    def load_state_dict(self, states):
        self.flow.load_state_dict(states)
         
    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.flow.parameters(),
                                self.clip)



class model(object):
    
    patience = 30
    
    def __init__(self, args, filename):
        
        self.__dict__.update(args.__dict__)

        self.filename = filename
        self.args = args        
        if args.dataset == 'power':
            p = 6
            D = load_maf_data('power')
        elif args.dataset == 'gas':
            p = 8
            D = load_maf_data('gas')
        elif args.dataset == 'hepmass':
            p = 21
            D = load_maf_data('hepmass')
        elif args.dataset == 'miniboone':
            p = 43
            D = load_maf_data('miniboone')
        elif args.dataset == 'bsds300':
            p = 63
            D = load_maf_data('bsds300')
        
        tr, va, te = D.trn.x, D.val.x, D.tst.x
            
            
        self.train_loader = data.DataLoader(tr, 
                                            batch_size=args.batch_size,
                                            shuffle=True)
        self.valid_loader = data.DataLoader(va, 
                                            batch_size=args.batch_size,
                                            shuffle=False)
        self.test_loader = data.DataLoader(te, 
                                           batch_size=args.batch_size,
                                           shuffle=False)
        
        self.maf = MAF(args, p)
        
        # optim
        amsgrad = bool(args.amsgrad)
        polyak = args.polyak
        self.optim = optim.Adam(self.maf.flow.parameters(),
                                lr=args.lr, 
                                betas=(args.beta1, args.beta2),
                                amsgrad=amsgrad,
                                polyak=polyak)
        
        
        # initialize checkpoint
        self.checkpoint = dict()
        self.checkpoint['best_val'] = float('inf')
        self.checkpoint['best_val_epoch'] = 0
        self.checkpoint['e'] = 0
    
     
    def train(self, epoch):
        optim = self.optim
        t = 0 
        
        LOSSES = 0
        counter = 0
        
        #for e in range(epoch):
        while self.checkpoint['e'] < epoch:
            for x in self.train_loader:
                optim.zero_grad()
                x = Variable(x)
                if self.cuda:
                    x = x.cuda()
                    
                losses = self.maf.loss(x)
                
                loss = losses.mean()
                
                LOSSES += losses.sum().data.cpu().numpy()
                counter += losses.size(0)

                loss.backward()
                self.maf.clip_grad_norm()
                optim.step()
                t += 1
                
                

            if self.checkpoint['e']%1 == 0:      
                optim.swap()
                loss_val = self.evaluate(self.valid_loader)
                loss_tst = self.evaluate(self.test_loader)       
                print 'Epoch: [%4d/%4d] train <= %.2f ' \
                      'valid: %.3f test: %.3f' % \
                (self.checkpoint['e']+1, epoch, LOSSES/float(counter), 
                 loss_val,
                 loss_tst)
                if loss_val < self.checkpoint['best_val']:
                    print(' [^] Best validation loss [^] ... [saving]')
                    self.save(self.save_dir+'/'+self.filename+'_best')
                    self.checkpoint['best_val'] = loss_val
                    self.checkpoint['best_val_epoch'] = self.checkpoint['e']+1
                    
                LOSSES = 0
                counter = 0
                optim.swap()
                
            self.checkpoint['e'] += 1
            if (self.checkpoint['e'])%5 == 0:
                self.save(self.save_dir+'/'+self.filename+'_last')
            
            if self.impatient():
                print 'Terminating due to impatience ... \n'
                break 
            
        # loading best valid model (early stopping)
        self.load(self.save_dir+'/'+self.filename+'_best')

    def impatient(self):
        current_epoch = self.checkpoint['e']
        bestv_epoch = self.checkpoint['best_val_epoch']
        return current_epoch - bestv_epoch > self.patience
        
        
    def evaluate(self, dataloader):
        LOSSES = 0 
        c = 0
        for x in dataloader:
            x = Variable(x)
            if self.cuda:
                x = x.cuda()
                
            losses = self.maf.loss(x).data.cpu().numpy()
            LOSSES += losses.sum()
            c += losses.shape[0]
        return LOSSES / float(c)


    def save(self, fn):        
        torch.save(self.maf.state_dict(), fn+'_model.pt')
        torch.save(self.optim.state_dict(), fn+'_optim.pt')
        with open(fn+'_args.txt','w') as out:
            out.write(json.dumps(self.args.__dict__,indent=4))
        with open(fn+'_checkpoint.txt','w') as out:     
            out.write(json.dumps(self.checkpoint,indent=4))

    def load(self, fn):
        self.maf.load_state_dict(torch.load(fn+'_model.pt'))
        self.optim.load_state_dict(torch.load(fn+'_optim.pt'))
    
    
    def resume(self, fn):
        self.load(fn)
        self.checkpoint.update(
            json.loads(open(fn+'_checkpoint.txt','r').read()))
    

# =============================================================================
# main
# =============================================================================


"""parsing and configuration"""
def parse_args():
    desc = "MAF"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='miniboone', 
                        choices=['power',
                                 'gas',
                                 'hepmass',
                                 'miniboone',
                                 'bsds300'])
    parser.add_argument('--epoch', type=int, default=400, 
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, 
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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--amsgrad', type=int, default=0)
    parser.add_argument('--polyak', type=float, default=0.0)
    parser.add_argument('--cuda', type=bool, default=False)
    
    
    parser.add_argument('--dimh', type=int, default=100)
    parser.add_argument('--flowtype', type=str, default='affine')
    parser.add_argument('--num_flow_layers', type=int, default=5)
    parser.add_argument('--num_hid_layers', type=int, default=1)
    parser.add_argument('--num_ds_dim', type=int, default=16)
    parser.add_argument('--num_ds_layers', type=int, default=1)
    parser.add_argument('--fixed_order', type=bool, default=True,
                        help='Fix the made ordering to be the given order')
    
    
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
   

def args2fn(args):
    
    prefix_key_pairs = [
        ('', 'dataset'),
        ('e', 'epoch'),
        ('s', 'seed'),
        ('p', 'polyak'),
        ('h', 'dimh'),
        ('f', 'flowtype'),
        ('fl', 'num_flow_layers'),
        ('l', 'num_hid_layers'),
        ('dsdim','num_ds_dim'),
        ('dsl', 'num_ds_layers'),
    ]
    
    return '_'.join([p+str(args.__dict__[k]) for p, k in prefix_key_pairs])
    
    

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed+10000)

    #fn = str(time.time()).replace('.','')
    fn = args2fn(args)
    print args
    print '\nfilename: ', fn

    print(" [*] Building model!")    
    if args.fn != '0':
        # overwrite
        # args.fn ends with ``_last'' or ``_best''
        old_fn = args.fn
        overwrite_args = True
        print 'MANUALLY RESUMING'
    else:
        # automatic resuming the last model
        # (of the same args) if it exists
        old_fn = fn + '_last'
        overwrite_args = False
        print 'AUTOMATICALLY RESUMING'
        
    old_args = args.save_dir+'/'+old_fn+'_args.txt'
    old_path = args.save_dir+'/'+old_fn
    if os.path.isfile(old_args):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        d = without_keys(json.loads(open(old_args,'r').read()),
                         ['to_train','epoch'])
        args.__dict__.update(d)
        if overwrite_args:
            fn = args2fn(args)
        print(" New args:" )
        print args
        print '\nfilename: ', fn
        mdl = model(args, fn)
        print(" [*] Loading model!")
        mdl.resume(old_path)
    else:
        mdl = model(args, fn)
    
    # launch the graph in a session
    if args.to_train:
        print(" [*] Training started!")
        mdl.train(args.epoch)
        print(" [*] Training finished!")

    print " [**] Valid: %.4f" % mdl.evaluate(mdl.valid_loader)
    print " [**] Test: %.4f" % mdl.evaluate(mdl.test_loader)

    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()

    
    
