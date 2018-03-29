#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:44:18 2018

@author: chinwei
"""



import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchkit import nn as nn_, flows, utils

import matplotlib.pyplot as plt


from scipy.stats import gaussian_kde


class model_1d(object):
    
    def __init__(self, target_energy):
        
        nd=24
        self.sf = flows.SigmoidFlow(num_ds_dim=nd)
        self.params = Parameter(torch.FloatTensor(1, 1, 3*nd).normal_())
        
        self.optim = optim.Adam([self.params,], lr=0.01, 
                                betas=(0.9, 0.999))
        
        self.target_energy = target_energy
        
    def sample(self, n):
        
        spl = Variable(torch.FloatTensor(n,1).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
       
        h, logdet = self.sf.forward(spl, lgd, self.params)
        out = nn_.sigmoid_(h) * 2.0 
        logdet = logdet + nn_.logsigmoid(h) + nn_.logsigmoid(-h) + np.log(2.0)
        return out, logdet
    def train(self):
        
        total = 20000
        
        for it in range(total):

            self.optim.zero_grad()
            w = min(1.0,0.2+it/float((total*0.80)))
            
            spl, logdet = self.sample(320)
            losses = - w * self.target_energy(spl) - logdet
            loss = losses.mean()
            
            loss.backward()
            self.optim.step()
            
            if ((it + 1) % 1000) == 0:
                print 'Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, loss.data[0])
             


# =============================================================================
# =============================================================================


# sine wave
class Sinewave(object):
    
    def __init__(self, a, f, phi):
        self.a = a
        self.f = f
        self.phi = phi
    
    def evaluate(self, t):
        # y(t) = a sin(2pi f t + phi)
        return self.a * np.sin(2*np.pi*self.f*t + self.phi)

xx = np.linspace(0,2,1000)

fig = plt.figure()
ax = plt.subplot(111)

sw1 = Sinewave(1.0, 0.6, 0.0)
yy1 = sw1.evaluate(xx)
plt.plot(xx,yy1,':')

sw2 = Sinewave(1.0, 1.2, 0.0)
yy2 = sw2.evaluate(xx)
plt.plot(xx,yy2,'--')

sw3 = Sinewave(1.0, 1.8, 0.0)
yy3 = sw3.evaluate(xx)
plt.plot(xx,yy3)


a0 = 1.0
f0 = 0.6
b0 = 0.0

x0 = utils.varify(np.array([[0.0],[5/6.],[10/6.]]).astype('float32'))
y0 = torch.mul(torch.sin(x0*2.0*np.pi*f0+b0), a0)

plt.scatter(x0.data.numpy()[:,0],y0.data.numpy()[:,0], color=(0.8,0.4,0.4),
            marker='x',s=100)

plt.rc('font', family='serif')
plt.title(r'$y(t) = \sin(2\pi f t)$', fontsize=18)
leg = plt.legend(['$f$=0.6','$f$=1.2','$f$=1.8'],
                 loc=4, fontsize=20)


plt.xlabel('t', fontsize=15)
plt.ylabel('y(t)', fontsize=15)
plt.tight_layout()

plt.savefig('sinewave_fs.pdf',format='pdf')


# =============================================================================
# =============================================================================


# Define energy function
# inferring q(f|(xi,yi)_i=1^3)
zero = Variable(torch.FloatTensor(1).zero_())
def energy1(f):
    mu = torch.mul(torch.sin(x0.permute(1,0)*2.0*np.pi*f+b0), a0)
    return - ((mu-y0.permute(1,0))**2 * (1/0.25)).sum(1)
    ll = utils.log_normal(y0.permute(1,0),mu,zero).sum(1)
    return ll


# =============================================================================
# =============================================================================
        

# build and train
mdl = model_1d(energy1)
mdl.train()


# =============================================================================
# =============================================================================


n = 10000
# plot figure
fig = plt.figure()
        
ax = fig.add_subplot(111)
spl = mdl.sample(n)[0]
plt.hist(spl.data.numpy()[:,0],100)
plt.xlabel('f', fontsize=15)
ax.set_yticklabels([])
plt.title('$f\sim q(f)$', fontsize=18)
plt.grid()
#plt.savefig('sinewave_qf.pdf',format='pdf')

spl = spl.data.numpy()
mdl_density = gaussian_kde(spl[:,0],0.05)
xx = np.linspace(0,2,1000)

plt.plot(xx,300*mdl_density(xx),'r')
plt.tight_layout()
plt.legend(['kde', 'counts'], loc=2, fontsize=20)
plt.savefig('sinewave_qf+kde.pdf',format='pdf')







