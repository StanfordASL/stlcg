import sys
sys.path.append('../../../src')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import stlcg
import matplotlib 


matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)


obs_1 = torch.tensor([0.0, 0.9, -1.0, -0.5]).float()
obs_2 = torch.tensor([.2, 0.7, 0.8, 1.2]).float()
obs_3 = torch.tensor([0.0, 0.0, 0.4]).float()
obs_4 = torch.tensor([-1.0, -0.7, -0.2, 0.5]).float()

x0 = -np.ones(2)
xf = np.ones(2)
Δt = 0.1
A = np.eye(2)
B = np.eye(2) * Δt
u_max = torch.as_tensor(0.8).float()

def generate_combined_always_plot(margin):
    linewidth = 10
    markersize = 100

    plt.figure(figsize=(10,10))
    plt.plot([obs_1[0], obs_1[0], obs_1[1], obs_1[1], obs_1[0]], [obs_1[2], obs_1[3], obs_1[3], obs_1[2], obs_1[2]], c="red", linewidth=5)
    plt.plot([obs_2[0], obs_2[0], obs_2[1], obs_2[1], obs_2[0]], [obs_2[2], obs_2[3], obs_2[3], obs_2[2], obs_2[2]], c="green", linewidth=5)
    plt.plot([obs_4[0], obs_4[0], obs_4[1], obs_4[1], obs_4[0]], [obs_4[2], obs_4[3], obs_4[3], obs_4[2], obs_4[2]], c="orange", linewidth=5)
    plt.plot([obs_3[0] + obs_3[2].numpy()*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [obs_3[1] + obs_3[2].numpy()*np.sin(t) for t in np.arange(0,3*np.pi,0.1)], c="blue", linewidth=5)



    u = np.load("../data/motion_planning/phi_2_margin=" + str(margin) + "/U.npy")
    u_norm = np.sqrt(np.sum(u**2, axis=1))
    u_norm = np.concatenate([u_norm, [u_norm[-1]]], axis=-1)
    alpha = u_norm/u_max.numpy()

    colors = np.ones([50,4])
    colors[:,:3] =  np.array([250, 128, 114])/255
    colors[:,1] = (127/255)* (alpha)

    xy = np.load("../data/motion_planning/phi_2_margin=" + str(margin) + "/X.npy")
    u = np.load("../data/motion_planning/phi_2_margin=" + str(margin) + "/U.npy")

    plt.plot(xy[:,0], xy[:,1], c="black", linewidth=linewidth)

    xs = [x0]
    us = u
    for i in range(49):
        x_next = A @ xs[i] + B @ us[i]
        xs.append(x_next)
    xs = np.stack(xs)
    plt.plot(xs[:,0], xs[:,1], c="SALMON")
    plt.scatter(xs[:,0], xs[:,1], s=markersize, marker='o', color=colors, zorder=10)


    u = np.load("../data/motion_planning/phi_1_margin=" + str(margin) + "/U.npy")
    u_norm = np.sqrt(np.sum(u**2, axis=1))
    u_norm = np.concatenate([u_norm, [u_norm[-1]]], axis=-1)
    alpha = u_norm/u_max.numpy()

    colors = np.ones([50,4])
    colors[:,:3] =  np.array([250, 128, 114])/255
    colors[:,1] = (127/255)* (alpha)

    xy = np.load("../data/motion_planning/phi_1_margin=" + str(margin) + "/X.npy")
    u = np.load("../data/motion_planning/phi_1_margin=" + str(margin) + "/U.npy")

    plt.plot(xy[:,0], xy[:,1], c="black", linewidth=linewidth)

    xs = [x0]
    us = u
    for i in range(49):
        x_next = A @ xs[i] + B @ us[i]
        xs.append(x_next)
    xs = np.stack(xs)
    plt.plot(xs[:,0], xs[:,1], c="salmon")
    plt.scatter(xs[:,0], xs[:,1], s=markersize, marker='o', color=colors, zorder=10)


    plt.axis("equal")
    plt.scatter([-1], [-1], s=300, c="orange", zorder=10)
    plt.scatter([1], [1], s=800, marker='*', c="orange", zorder=10)
    plt.text(-0.95, -1.1, "Start", fontsize=24)
    plt.text(0.9, 1.07, "Finish", fontsize=24)

    plt.text(0.55, 0.0, "$\phi_1$", fontsize=36)
    plt.text(-0.45, 0.45, "$\phi_2$", fontsize=36)
    plt.text(-0.97, 1.2, "Speed", fontsize=18)
    plt.text(-1.15, 1.05, "0", fontsize=18)
    plt.text(-0.6, 1.05, "$u_{max}$", fontsize=18)

    plt.text(-0.91, 0.1, "B1", fontsize=24)
    plt.text(0.37, -0.8, "B2", fontsize=24)
    plt.text(0.39, 0.96, "B3", fontsize=24)
    plt.text(-0.04, -0.04, "C", fontsize=24)


    colors = np.ones([50,4])
    colors[:,:3] =  np.array([250, 128, 114])/255
    colors[:,1] = (127/255)* np.arange(50)/49
    plt.scatter(np.arange(50)/49/2 - 1.1, 1.15*np.ones(50), color=colors, s=250, marker='s')

    plt.grid()
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.show()

def generate_combined_integral_plot(margin):
    linewidth = 10
    markersize = 100

    plt.figure(figsize=(10,10))
    plt.plot([obs_1[0], obs_1[0], obs_1[1], obs_1[1], obs_1[0]], [obs_1[2], obs_1[3], obs_1[3], obs_1[2], obs_1[2]], c="red", linewidth=5)
    plt.plot([obs_2[0], obs_2[0], obs_2[1], obs_2[1], obs_2[0]], [obs_2[2], obs_2[3], obs_2[3], obs_2[2], obs_2[2]], c="green", linewidth=5)
    plt.plot([obs_4[0], obs_4[0], obs_4[1], obs_4[1], obs_4[0]], [obs_4[2], obs_4[3], obs_4[3], obs_4[2], obs_4[2]], c="orange", linewidth=5)
    plt.plot([obs_3[0] + obs_3[2].numpy()*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [obs_3[1] + obs_3[2].numpy()*np.sin(t) for t in np.arange(0,3*np.pi,0.1)], c="blue", linewidth=5)



    u = np.load("../data/motion_planning/phi_2_integral_margin=" + str(margin) + "/U.npy")
    u_norm = np.sqrt(np.sum(u**2, axis=1))
    u_norm = np.concatenate([u_norm, [u_norm[-1]]], axis=-1)
    alpha = u_norm/u_max.numpy()

    colors = np.ones([50,4])
    colors[:,:3] =  np.array([250, 128, 114])/255
    colors[:,1] = (127/255)* (alpha)

    xy = np.load("../data/motion_planning/phi_2_integral_margin=" + str(margin) + "/X.npy")
    u = np.load("../data/motion_planning/phi_2_integral_margin=" + str(margin) + "/U.npy")

    plt.plot(xy[:,0], xy[:,1], c="black", linewidth=linewidth)

    xs = [x0]
    us = u
    for i in range(49):
        x_next = A @ xs[i] + B @ us[i]
        xs.append(x_next)
    xs = np.stack(xs)
    plt.plot(xs[:,0], xs[:,1], c="SALMON")
    plt.scatter(xs[:,0], xs[:,1], s=markersize, marker='o', color=colors, zorder=10)


    u = np.load("../data/motion_planning/phi_1_integral_margin=" + str(margin) + "/U.npy")
    u_norm = np.sqrt(np.sum(u**2, axis=1))
    u_norm = np.concatenate([u_norm, [u_norm[-1]]], axis=-1)
    alpha = u_norm/u_max.numpy()

    colors = np.ones([50,4])
    colors[:,:3] =  np.array([250, 128, 114])/255
    colors[:,1] = (127/255)* (alpha)

    xy = np.load("../data/motion_planning/phi_1_integral_margin=" + str(margin) + "/X.npy")
    u = np.load("../data/motion_planning/phi_1_integral_margin=" + str(margin) + "/U.npy")

    plt.plot(xy[:,0], xy[:,1], c="black", linewidth=linewidth)

    xs = [x0]
    us = u
    for i in range(49):
        x_next = A @ xs[i] + B @ us[i]
        xs.append(x_next)
    xs = np.stack(xs)
    plt.plot(xs[:,0], xs[:,1], c="salmon")
    plt.scatter(xs[:,0], xs[:,1], s=markersize, marker='o', color=colors, zorder=10)


    plt.axis("equal")
    plt.scatter([-1], [-1], s=300, c="orange", zorder=10)
    plt.scatter([1], [1], s=800, marker='*', c="orange", zorder=10)
    plt.text(-0.95, -1.1, "Start", fontsize=24)
    plt.text(0.9, 1.07, "Finish", fontsize=24)

    plt.text(0.55, 0.0, "$\psi_1$", fontsize=36)
    plt.text(-0.45, 0.45, "$\psi_2$", fontsize=36)
    plt.text(-0.97, 1.2, "Speed", fontsize=18)
    plt.text(-1.15, 1.05, "0", fontsize=18)
    plt.text(-0.6, 1.05, "$u_{max}$", fontsize=18)

    plt.text(-0.91, 0.1, "B1", fontsize=24)
    plt.text(0.37, -0.8, "B2", fontsize=24)
    plt.text(0.39, 0.96, "B3", fontsize=24)
    plt.text(-0.04, -0.04, "C", fontsize=24)

    colors = np.ones([50,4])
    colors[:,:3] =  np.array([250, 128, 114])/255
    colors[:,1] = (127/255)* np.arange(50)/49
    plt.scatter(np.arange(50)/49/2 - 1.1, 1.15*np.ones(50), color=colors, s=250, marker='s')

    plt.grid()
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.show()


if __name__ == '__main__':
    margin = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    generate_combined_integral_plot(margin)
    generate_combined_always_plot(margin)