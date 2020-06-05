import sys
sys.path.append('../../../src')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import stlcg

import scipy.io

mat = scipy.io.loadmat('../data/pSTL/clustering_data.mat')

ys_ = mat['ys']
ys = np.zeros(ys_.shape)
ys[:,:] = np.fliplr(ys_)
t = mat['T'][0,:]
N = ys.shape[0]

def run_multiple_times(num, func, ys):
    for i in range(num):
        func(ys, save_filename=None)

def binary_search_settling(ys, save_filename="../data/clustering_settling_binary_search.npy"):
    ϵ_list = []
    N = ys.shape[0]
    for i in range(N):
        y = torch.as_tensor(ys[i:i+1,:]).float().unsqueeze(-1)
        s = stlcg.Expression('s', torch.abs(y - 1))
        ϵL = torch.as_tensor(np.zeros([1, 1, 1])).float().requires_grad_(True)
        ϵU = torch.as_tensor(np.ones([1, 1, 1])).float().requires_grad_(True)
        ϵ = torch.as_tensor(np.ones([1, 1, 1])).float().requires_grad_(True)
        ϕL = stlcg.Always(subformula=(s < ϵL), interval=[50, 100])
        ϕU = stlcg.Always(subformula=(s < ϵU), interval=[50, 100])
        ϕ =  stlcg.Always(subformula=(s < ϵ), interval=[50, 100])
        while torch.abs(ϕU.subformula.val - ϕL.subformula.val) > 5*1E-3:
            ϵ = 0.5 * (ϕU.subformula.val + ϕL.subformula.val)
            ϕ.subformula.val = ϵ
            r = ϕ.robustness(s).squeeze()

            if r > 0:
                ϕU.subformula.val = ϵ
            else:
                ϕL.subformula.val = ϵ
        ϵ_list.append(ϵ.squeeze().detach().numpy())
    if save_filename is None:
        return
    np.save(save_filename, np.stack(ϵ_list))


def stlcg_settling(ys, save_filename="../data/clustering_settling_stlcg.npy"):
    max_epoch = 1000
    N = ys.shape[0]
    y = torch.as_tensor(ys).float().unsqueeze(-1)
    s = stlcg.Expression('s', torch.abs(y - 1))
    ϵ = torch.as_tensor(np.zeros([N, 1, 1])).float().requires_grad_(True)
    ϕ = stlcg.Always(subformula=(s < ϵ), interval=[50, 100])

    for epoch in range(max_epoch):

        loss = torch.relu(-ϕ.robustness(s).squeeze()).sum()
        loss.backward()

        with torch.no_grad():
            ϵ -= 0.005* ϵ.grad

        ϵ.grad.zero_()
        if loss == 0:
            break
    if save_filename is None:
        return
    np.save(save_filename, ϵ.squeeze().detach().numpy())


def binary_search_overshoot(ys, save_filename="../data/clustering_overshoot_binary_search.npy"):
    ϵ_list = []
    N = ys.shape[0]
    for i in range(N):
        y = torch.as_tensor(ys[i:i+1,:]).float().unsqueeze(-1)
        s = stlcg.Expression('s', y)
        ϵL = torch.as_tensor(np.zeros([1, 1, 1])).float().requires_grad_(True)
        ϵU = torch.as_tensor(2*np.ones([1, 1, 1])).float().requires_grad_(True)
        ϵ = torch.as_tensor(np.ones([1, 1, 1])).float().requires_grad_(True)
        ϕL = stlcg.Always(subformula=(s < ϵL))
        ϕU = stlcg.Always(subformula=(s < ϵU))
        ϕ =  stlcg.Always(subformula=(s < ϵ))
        while torch.abs(ϕU.subformula.val - ϕL.subformula.val) > 5*1E-3:
            ϵ = 0.5 * (ϕU.subformula.val + ϕL.subformula.val)
            ϕ.subformula.val = ϵ
            r = ϕ.robustness(s).squeeze()

            if r > 0:
                ϕU.subformula.val = ϵ
            else:
                ϕL.subformula.val = ϵ
        ϵ_list.append(ϵ.squeeze().detach().numpy())
    if save_filename is None:
        return
    np.save(save_filename, np.stack(ϵ_list))

def stlcg_overshoot(ys, save_filename="../data/clustering_overshoot_stlcg.npy"):
    N = ys.shape[0]
    max_epoch = 1000
    y = torch.as_tensor(ys).float().unsqueeze(-1)
    s = stlcg.Expression('s', y)
    ϵ = torch.as_tensor(np.zeros([N, 1, 1])).float().requires_grad_(True)
    ϕ = stlcg.Always(subformula=(s < ϵ))
    for epoch in range(max_epoch):

        loss = torch.relu(-ϕ.robustness(s).squeeze()).sum()
        loss.backward()

        with torch.no_grad():
            ϵ -= 0.005* ϵ.grad
        if loss == 0:
            break
        ϵ.grad.zero_()

    if save_filename is None:
        return
    np.save(save_filename, ϵ.squeeze().cpu().detach().numpy())


def stlcg_gpu_settling(ys, save_filename="../data/clustering_settling_stlcg.npy"):
    max_epoch = 1000
    N = ys.shape[0]
    y = torch.as_tensor(ys).float().unsqueeze(-1).cuda()
    s = stlcg.Expression('s', torch.abs(y - 1).cuda())
    ϵ = torch.as_tensor(np.zeros([N, 1, 1])).float().cuda().requires_grad_(True)
    ϕ = stlcg.Always(subformula=(s < ϵ).cuda(), interval=[50, 100]).cuda()


    for epoch in range(max_epoch):

        loss = torch.relu(-ϕ.robustness(s).squeeze()).sum()
        loss.backward()

        with torch.no_grad():
            ϵ -= 0.005* ϵ.grad

        ϵ.grad.zero_()
        if loss == 0:
            break
    if save_filename is None:
        return
    np.save(save_filename, ϵ.squeeze().cpu().detach().numpy())

def stlcg_gpu_overshoot(ys, save_filename="../data/clustering_overshoot_stlcg.npy"):
    N = ys.shape[0]
    max_epoch = 1000
    y = torch.as_tensor(ys).float().unsqueeze(-1).cuda()
    s = stlcg.Expression('s', y)
    ϵ = torch.as_tensor(np.zeros([N, 1, 1])).float().cuda().requires_grad_(True)
    ϕ = stlcg.Always(subformula=(s < ϵ)).cuda()
    for epoch in range(max_epoch):

        loss = torch.relu(-ϕ.robustness(s).squeeze()).sum()
        loss.backward()

        with torch.no_grad():
            ϵ -= 0.005* ϵ.grad
        if loss == 0:
            break
        ϵ.grad.zero_()

    if save_filename is None:
        return
    np.save(save_filename, ϵ.squeeze().cpu().detach().numpy())

def temperature_schedule(i, max_iter, a=15, b=10, c=2):
    t = np.arange(0, 1, 1.0/max_iter)
    return b*np.exp(a*t[i] - a/c)/(1 + np.exp(a*t[i] - a/c))


def stlcg_gpu_trafficweave(n_trials):
    d = np.load("/data/trafficweave/trajectories_slim.npz")
    n_samples = 30
    # n_trials = 100
    clip_val = 9
    max_tl = np.max(d['traj_lengths'][:n_trials])
    y_mat = np.zeros([n_trials, max_tl, 1])
    tco_mat = np.zeros([n_trials, max_tl, 1])
    for i in range(n_trials):
        car = 'car1' if d['car1'][i,0,1] > -4 else 'car2'
        y_mat[i,-d['traj_lengths'][i]:,:] = d[car][i:i+1,:d['traj_lengths'][i],1:2]
        tco_mat[i,-d['traj_lengths'][i]:,:] = d['extras'][i:i+1,:d['traj_lengths'][i],0:1]
    tco_mat = np.clip(tco_mat, -clip_val, clip_val)

    y_samples = np.repeat(np.expand_dims(y_mat, axis=0), n_samples, axis=0)
    y_samples = np.reshape(np.transpose(y_samples, [1,0,2,3]), [-1, y_mat.shape[1], 1])
    tco_samples = np.repeat(np.expand_dims(tco_mat, axis=0), n_samples, axis=0)
    tco_samples = np.reshape(np.transpose(tco_samples, [1,0,2,3]), [-1, tco_mat.shape[1], 1])
    tco_cutoff_np = np.expand_dims(np.reshape(np.repeat(np.expand_dims(np.linspace(-10, 0.0, n_samples), axis=0), n_trials, axis=0), [-1,1]), -1)


    y = stlcg.Expression('y', torch.as_tensor(y_samples).float().requires_grad_(False).flip(1).cuda())
    tco = stlcg.Expression('tco', torch.as_tensor(tco_samples).float().requires_grad_(False).flip(1).cuda())
    mid_lane = torch.tensor(-4, dtype=torch.float, requires_grad=False).cuda()

    tco_cutoff = torch.as_tensor(tco_cutoff_np).float().cuda().requires_grad_(True)
    tco_cutoff_static = torch.as_tensor(tco_cutoff_np).float().cuda().requires_grad_(False)

    in_lane = (y > mid_lane).cuda()
    tco_formula = (tco < tco_cutoff).cuda()
    tco_formula_static = (tco < tco_cutoff_static).cuda()
    change_lane_tco = stlcg.Until(subformula1=in_lane, subformula2=tco_formula).cuda()
    change_lane_tco_static = stlcg.Until(subformula1=in_lane, subformula2=tco_formula_static).cuda()
    inputs = (y, tco)

    tl_mat = torch.arange(1, max_tl+1).unsqueeze(0).repeat(n_samples*n_trials, 1).cuda()
    tls = torch.as_tensor(np.repeat(d['traj_lengths'][:n_trials], n_samples)).float().unsqueeze(1).cuda()

    iter_max = 1500
    for j in range(iter_max):
        inputs = (y, tco)
        scale = temperature_schedule(j, iter_max, b=10)+1
        loss = torch.masked_select(change_lane_tco(inputs, scale=scale).squeeze(), tl_mat == tls).pow(2).sum()
        loss.backward()
        if tco_cutoff.grad.norm() < 1E-3:
            print("converged! %i", j)
            break
        with torch.no_grad():
            tco_cutoff -= 0.01 * tco_cutoff.grad
            tco_cutoff.grad.zero_()


if __name__ == "__main__":
    arguments = sys.argv[1:]
    M = 500  # batch size
    num = 1  # number of repetitions
    func_i = int(sys.argv[1])
    func_map = {1: stlcg_settling, 2: binary_search_settling, 3: stlcg_overshoot, 4: binary_search_overshoot, 5: stlcg_gpu_settling, 6: stlcg_gpu_overshoot}
    if len(arguments) == 2:
        M = int(sys.argv[2])
    elif len(arguments) == 3:
        M = int(sys.argv[2])
        num = int(sys.argv[3])
    # print(func_map[func_i].__name__, "  M=%i   num=%i"%(M,num))

    mat = scipy.io.loadmat('../data/pSTL/clustering_data.mat')

    ys_ = np.concatenate([mat['ys'], mat['ys']], axis=0)[:M,:]
    ys = np.zeros(ys_.shape)
    ys[:,:] = np.fliplr(ys_)
    run_multiple_times(num, func_map[func_i], ys)


