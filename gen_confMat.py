#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:30:50 2018

@author: javed
"""
 
import numpy as np
import matplotlib.pyplot as  plt
import scipy.io
#import seaborn as sns
import math 
cm = scipy.io.loadmat('infrared_cm_mvit.mat')

norm_conf = cm['confMat']
conf_arr = cm['confMat']
fs1 = 22
fs2 = 19

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
#ax.xaxis.set_ticks_position('top')
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.YlGn, 
                interpolation='none')

width, height = norm_conf.shape

thresh = conf_arr.max() / 2.

for x in range(width):
    flag=0
    for y in range(height):   

        if conf_arr[x][y] != 0:
            ax.annotate(str(norm_conf[x][y])+'%', xy=(y, x),
                        horizontalalignment='center', fontsize=fs2,
                        verticalalignment='center', color="white" if conf_arr[x, y] > thresh else "black")

#cb = fig.colorbar(res)
labels = ('fight','handclap','handshake','hug','jog' ,'jump','punch', 'push', 'skip', 'walk', 'wave1', 'wave2')
plt.xticks([], [])

plt.yticks(range(height), labels, fontsize=fs1)
plt.show()
