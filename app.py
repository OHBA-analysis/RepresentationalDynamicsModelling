#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MWJ van Es, 2022
"""


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# define standard parameters
t=np.arange(0.001, 0.101, 0.001)

#%% example 1
f1 = 10 # frequency of signal 1
a1 = 1.5 # magnitude of signal 1, channel 1
a2 = a1*0.5 # magnitude of signal 1, channel 2
a = [a1, a2]
s1 = 0.5 # noise magnitude channel 1
s2 = 0.5 # noise magnitude channel 2
s = [s1, s2]
theta_a1 = -np.pi/2
theta_a2 = -np.pi/2
theta = [theta_a1, theta_a2]

# signals
xa=[]
x0=[]
for k in range(2):
    xa.append(a[k]*np.cos(2*np.pi*f1*t + theta[k])) # signal of condition 1, channel k
    x0.append(0*t)


# compute mutual information
A = np.diag(a)
Sigma = np.diag(s)**2
c_om = 0.5 * np.trace(A * np.linalg.inv(Sigma) * A * np.cos(theta-np.transpose(theta)))
temp1 = 0.5 * np.trace(A * np.linalg.inv(Sigma) * A * np.sin(theta+np.transpose(theta)))
temp2 = 0.5 * np.trace(A * np.linalg.inv(Sigma) * A * np.cos(theta+np.transpose(theta)))
r_om = np.sqrt(temp1**2 + temp2**2)
psi_om = np.arctan2(temp1, temp2)
alphaterm = c_om + r_om * np.cos(2*np.pi*2*f1*t + psi_om)

a.append(c_om)

grid_time = np.linspace(0,0.1,9)
grid_amp = np.linspace(-2,2,21)
grid_pow = np.linspace(0,2,21)
grid_freq = np.linspace(0,40,11)
grid_mi = np.linspace(0, 12, 21)
grid_mipow = np.linspace(0, 7, 21)

ticks_time = np.linspace(0,0.1,3)
ticks_amp = np.linspace(-2,2,5)
ticks_freq = np.linspace(0,40,3)
ticks_pow = np.linspace(0,2,5)
ticks_mi = np.linspace(0, 12, 5)
ticks_mipow = np.linspace(0,7,5)
fig = plt.figure()
for k in range(3):
    ax1 = fig.add_subplot(3, 2, 2 * (k + 1) - 1)
    if k<2:
        f=f1
        linefmt = 'b'
        markerfmt = 'bo'
        plt.plot(t, x0[k], 'k', linewidth=2)
        plt.fill_between(t, x0[k]-s[k], x0[k]+s[k], facecolor='gray', alpha=0.3)
        plt.plot(t, xa[k], 'b', linewidth=2)
        plt.fill_between(t, xa[k]-s[k], xa[k]+s[k], facecolor='b', alpha=0.3)
        plt.ylim((-2, 2))
        ax1.set_aspect(1./ax1.get_data_ratio())
        ax1.set_xticks(grid_time, minor=True)
        ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k]+s[k])),np.ceil(np.max(xa[k]+s[k])),21), minor=True)
        ax1.set_xticks(ticks_time)
        ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k]+s[k])),np.ceil(np.max(xa[k]+s[k])),5))
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.tick_params(which='minor', bottom=False, left=False)
        plt.ylabel('Magnitude')
    else:
        plt.plot(t, alphaterm, linewidth=2, color='k')
        plt.xlim((0, np.max(t)))
        ax1.set_aspect(1./ax1.get_data_ratio())
        ax1.set_xticks(grid_time, minor=True)
        ax1.set_yticks(grid_mi, minor=True)
        ax1.set_xticks(ticks_time)
        ax1.set_yticks(ticks_mi)
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.tick_params(which='minor', bottom=False, left=False)
        plt.ylabel('f^-1 (I(X,Y))')

        f=2*f1
        linefmt = 'k'
        markerfmt = 'ko'

    plt.xlabel('Time')
    ax2 = fig.add_subplot(3,2,2*(k+1))
    markerline, stemlines, baseline = plt.stem(
        [f], [a[k]], linefmt=linefmt, markerfmt=markerfmt)
    markerline.set_markerfacecolor('none')
    if k<2:
        plt.ylim((0, 2))
        ax2.set_aspect(10*ax2.get_data_ratio())
        ax2.set_yticks(grid_pow, minor=True)
        ax2.set_yticks(ticks_pow)
    else:
        plt.ylim((0,7))
        ax2.set_aspect(1.65*ax2.get_data_ratio())
        ax2.set_yticks(grid_mipow, minor=True)
        ax2.set_yticks(ticks_mipow)
    ax2.set_xticks(grid_freq, minor=True)
    ax2.set_xticks(ticks_freq)
    ax2.grid(which='both', linestyle='--', linewidth=0.5)
    ax2.tick_params(which='minor', bottom=False, left=False)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.show()


#%% Example 2
# This example has multiple frequncy components:
f1 = 10.0
f2 = 1.5*f1
f = np.array([f1, f2])
a1a = 0.7*0.7 # magnitude of signal 1, channel 1, frequency component 1
a1b = 1.1*0.7 # magnitude of signal 1, channel 1, frequency component 2
a2a = 0.5*a1a # magnitude of signal 1, channel 2, frequency component 1
a2b = 1.5 # magnitude of signal 1, channel 2, frequency component 2
a = np.array(((a1a, a1b), (a2a, a2b)))
a01a = 0.9*0.2*a1a # magnitude of signal 2, channel 1, frequency component 1
a01b = 0.9*0.6 # magnitude of signal 2, channel 1, frequency component 1
a02a = a01a # magnitude of signal 2, channel 1, frequency component 1
a02b = a01b # magnitude of signal 2, channel 1, frequency component 1
a0 = np.array(((a01a, a01b), (a02a, a02b)))
s1 = 0.5 # noise magnitude channel 1
s2 = 0.5 # noise magnitude channel 2
s = np.array([s1, s2])
theta_a1a = -np.pi/2
theta_a1b = -np.pi/2
theta_a2a = -np.pi/2
theta_a2b = -np.pi/2
theta = np.array(((theta_a1a, theta_a1b), (theta_a2a, theta_a2b)))
theta_a01a = -np.pi/2
theta_a01b = -np.pi/2
theta_a02a = -np.pi/2
theta_a02b = -np.pi/2
theta0 = np.array(((theta_a01a, theta_a01b), (theta_a02a, theta_a02b)))


# signals
xa=[]
x0=[]
c_om=[]
for k in range(2):
    xa.append(a[k,0]*np.cos(2*np.pi*f[0]*t + theta[k,0]) + a[k,1]*np.cos(2*np.pi*f[1]*t + theta[k,1])) # first channel, first condition
    x0.append(a0[k,0]*np.cos(2*np.pi*f[0]*t + theta0[k,0]) + a0[k,1]*np.cos(2*np.pi*f[1]*t + theta0[k,1])) # first channel, second condition
    tmp = 0.5*np.trace(np.diag(a[:,k]) * np.linalg.inv(Sigma) * np.diag(a[:,k]) * np.cos(theta[:,k].reshape((2,1)) - theta[:,k].reshape((1,2))))
    c_om.append()

# compute MI
Sigma = np.diag(s)**2
alphaterm=[]
for k in range(len(t)):
    alphaterm.append(np.matmul(np.matmul(np.array(((xa[0][k]-x0[0][k]), (xa[1][k]-x0[1][k]))), np.linalg.inv(Sigma)), np.array(((xa[0][k]-x0[0][k]), (xa[1][k]-x0[1][k])))))

c_om = np.array([1.5,1,0.3,0.4])


grid_time = np.linspace(0,0.1,9)
grid_amp = np.linspace(-2,2,21)
grid_pow = np.linspace(0,2,21)
grid_freq = np.linspace(0,40,11)
grid_mi = np.linspace(0, 12, 21)
grid_mipow = np.linspace(0, 7, 21)

ticks_time = np.linspace(0,0.1,3)
ticks_amp = np.linspace(-2,2,5)
ticks_freq = np.linspace(0,40,3)
ticks_pow = np.linspace(0,2,5)
ticks_mi = np.linspace(0, 12, 5)
ticks_mipow = np.linspace(0,7,5)
fig = plt.figure()
for k in range(3):
    ax1 = fig.add_subplot(3, 2, 2 * (k + 1) - 1)
    if k<2:
        linefmt = 'b'
        markerfmt = 'bo'
        plt.plot(t, x0[k], 'k', linewidth=2)
        plt.fill_between(t, x0[k]-s[k], x0[k]+s[k], facecolor='gray', alpha=0.3)
        plt.plot(t, xa[k], 'b', linewidth=2)
        plt.fill_between(t, xa[k]-s[k], xa[k]+s[k], facecolor='b', alpha=0.3)
        plt.ylim((-2, 2))
        ax1.set_box_aspect(1)
        ax1.set_xticks(grid_time, minor=True)
        ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k]+s[k])),np.ceil(np.max(xa[k]+s[k])),21), minor=True)
        ax1.set_xticks(ticks_time)
        ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k]+s[k])),np.ceil(np.max(xa[k]+s[k])),5))
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.tick_params(which='minor', bottom=False, left=False)
        plt.ylabel('Magnitude')
    else:
        plt.plot(t, alphaterm, linewidth=2, color='k')
        plt.xlim((0, np.max(t)))
        ax1.set_box_aspect(1)
        ax1.set_xticks(grid_time, minor=True)
        ax1.set_yticks(grid_mi, minor=True)
        ax1.set_xticks(ticks_time)
        ax1.set_yticks(ticks_mi)
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.tick_params(which='minor', bottom=False, left=False)
        plt.ylabel('f^-1 (I(X,Y))')

        linefmt = 'k'
        markerfmt = 'ko'

    plt.xlabel('Time')
    ax2 = fig.add_subplot(3,2,2*(k+1))
    if k<2:
        markerline, stemlines, baseline = plt.stem(
            2 * f, a[k, :], linefmt=linefmt, markerfmt=markerfmt, basefmt='w')
        plt.ylim((0, 2))
        ax2.set_box_aspect(1)
        ax2.set_yticks(grid_pow, minor=True)
        ax2.set_yticks(ticks_pow)
    else:
        markerline, stemlines, baseline = plt.stem(
            np.concatenate((2*f, np.array((np.abs(np.diff(f))[0], np.abs(np.sum(f)))))), c_om, linefmt=linefmt, markerfmt=markerfmt, basefmt='w')
        plt.ylim((0,7))
        ax2.set_box_aspect(1)
        ax2.set_yticks(grid_mipow, minor=True)
        ax2.set_yticks(ticks_mipow)
    markerline.set_markerfacecolor('none')
    ax2.set_xticks(grid_freq, minor=True)
    ax2.set_xticks(ticks_freq)
    ax2.grid(which='both', linestyle='--', linewidth=0.5)
    ax2.tick_params(which='minor', bottom=False, left=False)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.show()





