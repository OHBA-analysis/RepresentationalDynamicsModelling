#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MWJ van Es, 2022
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# define standard parameters
t = np.arange(0.001, 0.101, 0.001)
example = 2
if example == 1:
    f1 = 10.0
    f2 = 0
    a1a = 1.5  # magnitude of signal 1, channel 1, frequency component 1
    a1b = 0  # magnitude of signal 1, channel 1, frequency component 2
    a2a = 0.5 * a1a  # magnitude of signal 1, channel 2, frequency component 1
    a2b = 0  # magnitude of signal 1, channel 2, frequency component 2
    a01a = 0  # magnitude of signal 2, channel 1, frequency component 1
    a01b = 0  # magnitude of signal 2, channel 1, frequency component 1
    a02a = 0  # magnitude of signal 2, channel 1, frequency component 1
    a02b = 0  # magnitude of signal 2, channel 1, frequency component 1
    s1 = 0.5  # noise magnitude channel 1
    s2 = 0.5  # noise magnitude channel 2
    theta_a1a = -np.pi / 2
    theta_a1b = -np.pi / 2
    theta_a2a = -np.pi / 2
    theta_a2b = -np.pi / 2
    theta_a01a = -np.pi / 2
    theta_a01b = -np.pi / 2
    theta_a02a = -np.pi / 2
    theta_a02b = -np.pi / 2

    ticks_amp = np.linspace(-2, 2, 5)
    ticks_pow = np.linspace(0, 2, 5)
    ticks_mi = np.linspace(0, 6, 4)
    ticks_mipow = np.linspace(0, 3, 4)

    grid_time = np.linspace(0, 0.1, 9)
    grid_amp = np.linspace(-2, 2, 21)
    grid_pow = np.linspace(0, 2, 21)
    grid_freq = np.linspace(0, 40, 11)
    grid_mi = np.linspace(0, 12, 21)
    grid_mipow = np.linspace(0, 7, 21)

    ylim0 = (0, 2)
    ylim1 = (0, 6)
    ylim2 = (0, 3)
elif example == 2:
    # This example has multiple frequncy components:
    f1 = 10.0
    f2 = 1.5 * f1
    a1a = 0.5  # magnitude of signal 1, channel 1, frequency component 1
    a1b = 0.77  # magnitude of signal 1, channel 1, frequency component 2
    a2a = 0.5 * a1a  # magnitude of signal 1, channel 2, frequency component 1
    a2b = 1.5  # magnitude of signal 1, channel 2, frequency component 2
    a01a = 0.09  # magnitude of signal 2, channel 1, frequency component 1
    a01b = 0.9 * 0.6  # magnitude of signal 2, channel 1, frequency component 1
    a02a = a01a  # magnitude of signal 2, channel 1, frequency component 1
    a02b = a01b  # magnitude of signal 2, channel 1, frequency component 1
    s1 = 0.5  # noise magnitude channel 1
    s2 = 0.5  # noise magnitude channel 2
    theta_a1a = -np.pi / 2
    theta_a1b = -np.pi / 2
    theta_a2a = -np.pi / 2
    theta_a2b = -np.pi / 2
    theta_a01a = -np.pi / 2
    theta_a01b = -np.pi / 2
    theta_a02a = -np.pi / 2
    theta_a02b = -np.pi / 2

    ticks_amp = np.linspace(-2, 2, 5)
    ticks_pow = np.linspace(0, 2, 5)
    ticks_mi = np.linspace(0, 1.5, 4)
    ticks_mipow = np.linspace(0, 0.5, 6)

    grid_time = np.linspace(0, 0.1, 9)
    grid_amp = np.linspace(-2, 2, 21)
    grid_pow = np.linspace(0, 2, 21)
    grid_freq = np.linspace(0, 40, 11)
    grid_mi = np.linspace(0, 12, 21)
    grid_mipow = np.linspace(0, 7, 21)

    ylim0 = (0, 2)
    ylim1 = (0, 1.5)
    ylim2 = (0, 0.5)
else:
    print('this is the interactive part')

ticks_time = np.linspace(0, 0.1, 3)
ticks_freq = np.linspace(0, 40, 3)

# combine parameters in arrays
a = np.array(((a1a, a1b), (a2a, a2b)))
a0 = np.array(((a01a, a01b), (a02a, a02b)))
f = np.array([f1, f2])
s = np.array([s1, s2])
Sig = np.diag(s) ** 2
theta = np.array(((theta_a1a, theta_a1b), (theta_a2a, theta_a2b)))
theta0 = np.array(((theta_a01a, theta_a01b), (theta_a02a, theta_a02b)))

# signals
xa = []
x0 = []
for k in range(2):
    xa.append(np.matmul(np.expand_dims(a[k, :], 0), np.cos(
        2 * np.pi * np.expand_dims(f, 1) * np.transpose(np.expand_dims(t, 1)) + np.expand_dims(theta[k, :], 1)))[
                  0])  # first channel, first condition
    x0.append(np.matmul(np.expand_dims(a0[k, :], 0), np.cos(
        2 * np.pi * np.expand_dims(f, 1) * np.transpose(np.expand_dims(t, 1)) + np.expand_dims(theta0[k, :], 1)))[0])

# now we need to map these two channel signals back to the format given in the paper:
A_omega = []
mu_omega = []
phi_omega = []
phi_mean = []
for ifreq in np.arange(2):
    A_omega.append(0.5 * np.diag(np.sqrt(a[:, ifreq] ** 2 + a0[:, ifreq] ** 2 +
                                         2 * a[:, ifreq] * a0[:, ifreq] * np.cos(
        theta[:, ifreq] - theta0[:, ifreq] + np.array((-np.pi, -np.pi))))))
    mu_omega.append(0.5 * np.diag(np.sqrt(a[:, ifreq] ** 2 + a0[:, ifreq] ** 2 +
                                          2 * a[:, ifreq] * a0[:, ifreq] * np.cos(theta[:, ifreq] - theta0[:, ifreq]))))
    phi_omega.append(np.arctan2(a[:, ifreq] * np.sin(theta[:, ifreq]) + a0[:, ifreq] * np.sin(theta0[:, ifreq] + np.pi),
                                a[:, ifreq] * np.cos(theta[:, ifreq]) + a0[:, ifreq] * np.cos(
                                    theta0[:, ifreq] + np.pi)))
    phi_mean.append(-np.pi / 2 + np.arctan2(
        a[:, ifreq] * np.sin(theta[:, ifreq] + np.pi / 2) + a0[:, ifreq] * np.sin(theta0[:, ifreq] + np.pi / 2),
        a[:, ifreq] * np.cos(theta[:, ifreq] + np.pi / 2) + a0[:, ifreq] * np.cos(theta0[:, ifreq] + np.pi / 2)))

# find the information content terms:
c_b = 0
if len(f[f > 0]) == 1:
    Sigma = Sig
else:
    Sigma = Sig + Sig  # broadband noise: sum over both frequency bands
r_b = []
psi = []
for ifreq in np.arange(2):
    c_b = c_b + np.trace(np.matmul(np.matmul(A_omega[ifreq], np.linalg.inv(Sigma)), A_omega[ifreq]) * np.cos(
        np.expand_dims(phi_omega[ifreq], 1) - phi_omega[ifreq]))  # equal to above expression
    temp1 = np.trace(np.matmul(np.matmul(A_omega[ifreq], np.linalg.inv(Sigma)),
                               A_omega[ifreq] * np.cos(np.expand_dims(phi_omega[ifreq], 1) + phi_omega[ifreq])))
    temp2 = np.trace(np.matmul(np.matmul(A_omega[ifreq], np.linalg.inv(Sigma)),
                               A_omega[ifreq] * np.sin(np.expand_dims(phi_omega[ifreq], 1) + phi_omega[ifreq])))
    r_b.append(np.sqrt(temp1 ** 2 + temp2 ** 2))
    psi.append(np.arctan2(temp2, temp1))

# and cross-frequency components:
posmin = [1, -1]
for i in np.arange(2):
    temp1 = np.trace(np.matmul(np.matmul(A_omega[0], np.linalg.inv(Sigma)),
                               A_omega[1] * np.cos(np.expand_dims(phi_omega[0], 1) + posmin[i] * phi_omega[1])))
    temp2 = np.trace(np.matmul(np.matmul(A_omega[0], np.linalg.inv(Sigma)),
                               A_omega[1] * np.sin(np.expand_dims(phi_omega[0], 1) + posmin[i] * phi_omega[1])))
    r_b.append(2 * np.sqrt(temp1 ** 2 + temp2 ** 2))
    psi.append(np.arctan2(temp2, temp1))
r_b = np.array(r_b)

infotermest = c_b
freqs_all = np.concatenate((2 * f, np.array([np.sum(f)]), np.diff(f)))
for ifreq in np.arange(4):
    infotermest = infotermest + r_b[ifreq] * np.cos(2 * np.pi * freqs_all[ifreq] * t + psi[ifreq])

# we do not need the alternative computation alphaterm = np.zeros((1,len(t))) for i in np.arange(len(t)): alphaterm[
# 0][i] = 2*np.matmul(np.matmul(np.expand_dims(np.array(((xa[0][i]-x0[0][i])/2, (xa[1][i]-x0[1][i])/2)),0),
# np.linalg.inv(Sigma)), np.expand_dims(np.array(((xa[0][i] - x0[0][i]) / 2, (xa[1][i] - x0[1][i]) / 2)), 1))[0][0]

if not np.logical_or(example == 1, example == 2):
    ticks_amp = np.linspace(np.floor(np.min((np.min(xa[0]) - s[0], np.min(xa[1]) - s[1]))),
                            np.ceil(np.max((np.max(xa[0]) + s[0], np.max(xa[1]) + s[1]))),
                            np.diff((np.floor(np.min((np.min(xa[0]) - s[0], np.min(xa[1]) - s[1]))),
                                     np.ceil(np.max((np.max(xa[0]) + s[0], np.max(xa[1]) + s[1])))))[0] + 1)
    ticks_pow = np.linspace(0, np.max((np.max(a), np.max(a0))), 2 * np.max((np.max(a), np.max(a0))) + 1)
    ticks_mi = np.linspace(0, np.ceil(np.max(infotermest)), 2 * np.ceil(np.max(infotermest)) + 1)
    ticks_mipow = np.linspace(0, np.ceil(np.max(r_b)), 2 * np.ceil(np.max(r_b)) + 1)
#%% Plot everything
fig = plt.figure()
for k in range(3):
    ax1 = fig.add_subplot(3, 2, 2 * (k + 1) - 1)
    if k < 2:
        plt.plot(t, x0[k], 'k', linewidth=2)
        plt.fill_between(t, x0[k] - s[k], x0[k] + s[k], facecolor='gray', alpha=0.3)
        plt.plot(t, xa[k], 'b', linewidth=2)
        plt.fill_between(t, xa[k] - s[k], xa[k] + s[k], facecolor='b', alpha=0.3)
        ax1.set_box_aspect(1)
        ax1.set_xticks(grid_time, minor=True)
        ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k] + s[k])), np.ceil(np.max(xa[k] + s[k])), 21), minor=True)
        ax1.set_xticks(ticks_time)
        ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k] + s[k])), np.ceil(np.max(xa[k] + s[k])), 5))
        # ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.tick_params(which='minor', bottom=False, left=False)
        plt.ylabel('Magnitude')
        plt.ylim((-2, 2))
    else:
        plt.plot(t, np.squeeze(infotermest), linewidth=2, color='k')
        plt.xlim((0, np.max(t)))
        ax1.set_box_aspect(1)
        ax1.set_xticks(grid_time, minor=True)
        ax1.set_yticks(grid_mi, minor=True)
        ax1.set_xticks(ticks_time)
        ax1.set_yticks(ticks_mi)
        # ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.tick_params(which='minor', bottom=False, left=False)
        plt.ylabel('f^-1 (I(X,Y))')
        plt.ylim(ylim1)

    plt.xlabel('Time')
    ax2 = fig.add_subplot(3, 2, 2 * (k + 1))
    if k < 2:
        if np.any(a0[k, :] > 0):
            markerline, stemlines, baseline = plt.stem(
                f[a0[k, :] > 0], a0[k, a0[k, :] > 0], linefmt='k', markerfmt='ko', basefmt='w')
            markerline.set_markerfacecolor('none')
        if np.any(a[k, :] > 0):
            markerline, stemlines, baseline = plt.stem(
                f[a[k, :] > 0], a[k, a[k, :] > 0], linefmt='b', markerfmt='bo', basefmt='w')
            markerline.set_markerfacecolor('none')
        plt.ylim((0, 2))
        ax2.set_box_aspect(1)
        ax2.set_yticks(grid_pow, minor=True)
        ax2.set_yticks(ticks_pow)
    else:
        markerline, stemlines, baseline = plt.stem(
            freqs_all[r_b > 0], r_b[r_b > 0], linefmt='k', markerfmt='ko', basefmt='w')
        ax2.set_box_aspect(1)
        ax2.set_yticks(grid_mipow, minor=True)
        ax2.set_yticks(ticks_mipow)
        plt.ylim(ylim2)
        markerline.set_markerfacecolor('none')
    ax2.set_xticks(grid_freq, minor=True)
    ax2.set_xticks(ticks_freq)
    # ax2.grid(which='both', linestyle='--', linewidth=0.5)
    ax2.tick_params(which='minor', bottom=False, left=False)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.show()
