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

def generate_plot_from_data(margin):
    plt.figure(figsize=(10,10))
    plt.plot([obs_1[0], obs_1[0], obs_1[1], obs_1[1], obs_1[0]], [obs_1[2], obs_1[3], obs_1[3], obs_1[2], obs_1[2]], c="red", linewidth=5)
    plt.plot([obs_2[0], obs_2[0], obs_2[1], obs_2[1], obs_2[0]], [obs_2[2], obs_2[3], obs_2[3], obs_2[2], obs_2[2]], c="green", linewidth=5)
    plt.plot([obs_4[0], obs_4[0], obs_4[1], obs_4[1], obs_4[0]], [obs_4[2], obs_4[3], obs_4[3], obs_4[2], obs_4[2]], c="orange", linewidth=5)
    plt.plot([obs_3[0] + obs_3[2].numpy()*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [obs_3[1] + obs_3[2].numpy()*np.sin(t) for t in np.arange(0,3*np.pi,0.1)], c="blue", linewidth=5)

    xy = np.load("../data/motion_planning/phi_2_margin=" + str(margin) + "/X.npy")
    u = np.load("../data/motion_planning/phi_2_margin=" + str(margin) + "/U.npy")
    plt.plot(xy[:,0], xy[:,1], c="black")
    plt.scatter(xy[:,0], xy[:,1], c="black", s=150)

    xs = [x0]
    us = u
    for i in range(49):
        x_next = A @ xs[i] + B @ us[i]
        xs.append(x_next)
    xs = np.stack(xs)
    plt.plot(xs[:,0], xs[:,1], c="LIGHTSALMON")
    plt.scatter(xs[:,0], xs[:,1], c="LIGHTSALMON", s=50)


    xy = np.load("../data/motion_planning/phi_1_margin=" + str(margin) + "/X.npy")
    u = np.load("../data/motion_planning/phi_1_margin=" + str(margin) + "/U.npy")
    plt.plot(xy[:,0], xy[:,1], c="black")
    plt.scatter(xy[:,0], xy[:,1], c="black", s=150)

    xs = [x0]
    us = u
    for i in range(49):
        x_next = A @ xs[i] + B @ us[i]
        xs.append(x_next)
    xs = np.stack(xs)
    plt.plot(xs[:,0], xs[:,1], c="lightskyblue")
    plt.scatter(xs[:,0], xs[:,1], c="lightskyblue", s=50)


    plt.axis("equal")
    plt.scatter([-1], [-1], s=300, c="orange")
    plt.scatter([1], [1], s=800, marker='*', c="orange")

    plt.grid()
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.title("Margin=%.2f"%margin, fontsize=32)
    plt.savefig("../figs/motion_planning/traj_results_margin=" + str(margin) + ".png")
    plt.close()

    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    u = np.load("../data/motion_planning/phi_1_margin=" + str(margin) + "/U.npy")
    plt.plot(np.sqrt(np.sum(u**2, axis=1)), label="$\|u\|$", linewidth=5)
    plt.plot(u[:,0], label="$u_x$", linewidth=3)
    plt.plot(u[:,1], label="$u_y$", linewidth=3)
    plt.plot([0, u.shape[0]], [u_max.numpy(),u_max.numpy()], c="black")
    plt.title("Control sequence $\phi_1$, Margin=%.2f"%margin, fontsize=32)
    # plt.xlabel("Time step", fontsize=fs)
    plt.ylabel("Control", fontsize=20)
    plt.ylim([0, 0.85])
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.subplot(2,1,2)
    u = np.load("../data/motion_planning/phi_2_margin=" + str(margin) + "/U.npy")
    plt.plot(np.sqrt(np.sum(u**2, axis=1)), label="$\|u\|$", linewidth=5)
    plt.plot(u[:,0], label="$u_x$", linewidth=3)
    plt.plot(u[:,1], label="$u_y$", linewidth=3)
    plt.plot([0, u.shape[0]], [u_max.numpy(),u_max.numpy()], c="black")
    plt.title("Control sequence $\phi_2$, Margin=%.2f"%margin, fontsize=32)
    plt.xlabel("Time steps", fontsize=20)
    plt.ylabel("Control", fontsize=20)
    plt.ylim([0, 0.85])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../figs/motion_planning/control_results_margin=" + str(margin) + ".png")
    plt.close()


if __name__ == '__main__':
    margin = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    generate_plot_from_data(margin)