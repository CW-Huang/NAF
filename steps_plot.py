#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:02:49 2018

@author: chinwei
"""




import numpy as np
import matplotlib.pyplot as plt


delta = 0.00001
sigmoid = lambda x: 1/(1+np.exp(-x))
logit = lambda x: np.log(x+delta) - np.log(1-delta-x)

nc = 7



xx = np.linspace(-5,5,1000)

inters = np.linspace(0,1,nc+1)
ys = inters[1:-1]
ys_ = ys.copy()
ys_[-1] = 1.0
xs = logit(ys)
tril = np.tril(np.ones(nc-1))
w = np.linalg.inv(tril).dot(ys_)

def step(x):
    return ((x[:,None] >= xs[None,:]).astype(float) * w).sum(1)

# sigmoids
dists = np.abs(xs.reshape(-1,1) - xs.reshape(1,-1))
kappa = float('inf')
for i in range(dists.shape[0]-1):
    kappa = min(kappa, dists[i+1:,i].min())

tau = kappa / logit(1-1/float(2*nc))

def sigmoids(x):
    return (sigmoid((x[:,None] - xs[None,:])/tau) * w).sum(1)


plt.rc('font', family='serif')


fig = plt.figure(figsize=(5,4) )
ax = fig.add_subplot(1,1,1)

ax.plot(xx,step(xx),'b',lw=2)
ax.plot(xx,sigmoid(xx),'r:',lw=2)
ax.plot(xx,sigmoids(xx),'g--',lw=2)






plt.xlim(-5,5)
plt.ylim(-0.05,1.05)
plt.grid()

#plt.ytick


plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
    
plt.tight_layout()
plt.legend([r'$S_n^*$',r'$S$',r'$S_n$'],loc=2,fontsize=15)

plt.savefig('steps_cdf.pdf',format='pdf')
















