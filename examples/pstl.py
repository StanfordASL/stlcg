import sys
sys.path.append('../src')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import stlcg

import scipy.io

mat = scipy.io.loadmat('models/clustering_data.mat')

ys_ = mat['ys']
ys = np.zeros(ys_.shape)
ys[:,:] = np.fliplr(ys_)
t = mat['T'][0,:]
N = ys.shape[0]

def run_multiple_times(num, func, ys):
    for i in range(num):
        func(ys, save_filename=None)

def binary_search_settling(ys, save_filename="models/pstl_settling_binary_search.npy"):
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
        return np.stack(ϵ_list)
    np.save(save_filename, np.stack(ϵ_list))

def binary_search_settling_vectorize(ys):
    N = ys.shape[0]
    y = torch.as_tensor(ys).float().unsqueeze(-1)
    s = stlcg.Expression('s', torch.abs(y - 1))
    ϵL = torch.as_tensor(np.zeros([N, 1, 1])).float().requires_grad_(True)
    ϵU = torch.as_tensor(np.ones([N, 1, 1])).float().requires_grad_(True)
    ϵ = torch.as_tensor(np.ones([N, 1, 1])).float().requires_grad_(True)
    ϕL = stlcg.Always(subformula=(s < ϵL), interval=[50, 100])
    ϕU = stlcg.Always(subformula=(s < ϵU), interval=[50, 100])
    ϕ =  stlcg.Always(subformula=(s < ϵ), interval=[50, 100])
    while torch.abs(ϕU.subformula.val - ϕL.subformula.val).max() > 5*1E-3:
        ϵ = 0.5 * (ϕU.subformula.val + ϕL.subformula.val)
        ϕ.subformula.val = ϵ
        r = ϕ.robustness(s)
        ϕU.subformula.val = torch.where(torch.abs(ϕU.subformula.val - ϕL.subformula.val) > 5*1E-3, torch.where(r > 0, ϵ, ϕU.subformula.val), ϕU.subformula.val)
        ϕL.subformula.val = torch.where(torch.abs(ϕU.subformula.val - ϕL.subformula.val) > 5*1E-3, torch.where(r <= 0, ϵ, ϕL.subformula.val), ϕL.subformula.val)
    return ϵ.squeeze().detach().numpy()

def stlcg_settling(ys, save_filename="models/pstl_settling_stlcg.npy"):
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
        return ϵ.squeeze().detach().numpy()
    np.save(save_filename, ϵ.squeeze().detach().numpy())


def binary_search_overshoot(ys, save_filename="models/pstl_overshoot_binary_search.npy"):
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
        return np.stack(ϵ_list)
    np.save(save_filename, np.stack(ϵ_list))
    
def binary_search_overshoot_vectorize(ys):
    N = ys.shape[0]
    y = torch.as_tensor(ys).float().unsqueeze(-1)
    s = stlcg.Expression('s', y)
    ϵL = torch.as_tensor(np.zeros([N, 1, 1])).float().requires_grad_(True)
    ϵU = torch.as_tensor(2*np.ones([N, 1, 1])).float().requires_grad_(True)
    ϵ = torch.as_tensor(np.ones([N, 1, 1])).float().requires_grad_(True)
    ϕL = stlcg.Always(subformula=(s < ϵL))
    ϕU = stlcg.Always(subformula=(s < ϵU))
    ϕ =  stlcg.Always(subformula=(s < ϵ))
    while torch.abs(ϕU.subformula.val - ϕL.subformula.val).max() > 5*1E-3:
        ϵ = 0.5 * (ϕU.subformula.val + ϕL.subformula.val)
        ϕ.subformula.val = ϵ
        r = ϕ.robustness(s)
        ϕU.subformula.val = torch.where(torch.abs(ϕU.subformula.val - ϕL.subformula.val) > 5*1E-3, torch.where(r > 0, ϵ, ϕU.subformula.val), ϕU.subformula.val)
        ϕL.subformula.val = torch.where(torch.abs(ϕU.subformula.val - ϕL.subformula.val) > 5*1E-3, torch.where(r <= 0, ϵ, ϕL.subformula.val), ϕL.subformula.val)
    return ϵ.squeeze().detach().numpy()

def stlcg_overshoot(ys, save_filename="models/pstl_overshoot_stlcg.npy"):
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
        return ϵ.squeeze().cpu().detach().numpy()
    np.save(save_filename, ϵ.squeeze().cpu().detach().numpy())


def stlcg_gpu_settling(ys, save_filename="models/pstl_settling_stlcg_gpu.npy"):
    max_epoch = 1000
    N = ys.shape[0]
    y = torch.as_tensor(ys).float().unsqueeze(-1).cuda()
    s = stlcg.Expression('s', torch.abs(y - 1))
    ϵ = torch.as_tensor(np.zeros([N, 1, 1])).float().cuda().requires_grad_(True)
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
        return ϵ.squeeze().cpu().detach().numpy()
    np.save(save_filename, ϵ.squeeze().cpu().detach().numpy())

def stlcg_gpu_overshoot(ys, save_filename="models/pstl_overshoot_stlcg_gpu.npy"):
    N = ys.shape[0]
    max_epoch = 1000
    y = torch.as_tensor(ys).float().unsqueeze(-1).cuda()
    s = stlcg.Expression('s', y)
    ϵ = torch.as_tensor(np.zeros([N, 1, 1])).float().cuda().requires_grad_(True)
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
        return ϵ.squeeze().cpu().detach().numpy()
    np.save(save_filename, ϵ.squeeze().cpu().detach().numpy())



if __name__ == "__main__":
    arguments = sys.argv[1:]
    # func_i, M, num
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

    mat = scipy.io.loadmat('models/clustering_data.mat')

    ys_ = np.concatenate([mat['ys'], mat['ys'], mat['ys']], axis=0)[:M,:]
    ys = np.zeros(ys_.shape)
    ys[:,:] = np.fliplr(ys_)
    run_multiple_times(num, func_map[func_i], ys)


