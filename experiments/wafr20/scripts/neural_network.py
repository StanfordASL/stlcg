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

a = 0.48
b = 0.52
def generate_plot_from_data_bump(robustness_weight):
    ### bump example ###

    plt.figure(figsize=(17,5))
    plt.subplot(1,2,1)
    data = np.load("../data/neural_network/bump/robustness_weight=0.00/data.npy")
    xx = data[0,:]
    yy = data[1,:]
    y_pred = data[2,:]
    plt.plot(xx+2, y_pred, linewidth=3, label="Model")
    plt.plot(xx+2, yy, linewidth=1, label="Data")
    plt.plot([0,4],[a]*2, 'k--', linewidth=1)
    plt.plot([0,4],[b]*2, 'k--', linewidth=1)
    plt.ylim([-0.8, 0.8])
    plt.xlabel("Time steps", fontsize=20)
    plt.ylabel("s", fontsize=20)
    plt.legend(fontsize=20)
    plt.title("Without robustness regularization", fontsize=26)
    plt.grid()


    plt.subplot(1,2,2)
    data = np.load("../data/neural_network/bump/robustness_weight=%.2f/data.npy"%robustness_weight)
    xx = data[0,:]
    yy = data[1,:]
    y_pred = data[2,:]
    plt.plot(xx+2, y_pred, linewidth=3, label="Model")
    plt.plot(xx+2, yy, linewidth=1, label="Data")
    plt.plot([0,4],[a]*2, 'k--', linewidth=1)
    plt.plot([0,4],[b]*2, 'k--', linewidth=1)
    plt.ylim([-0.8, 0.8])
    plt.xlabel("Time steps", fontsize=20)
    plt.ylabel("s", fontsize=20)
    plt.legend(fontsize=20)
    plt.title("With robustness regularization $\gamma$=%.2f"%robustness_weight, fontsize=26)
    plt.grid()

    plt.tight_layout()

    plt.savefig("../figs/neural_network/bump/bump_comparison_robustness_weight=%.2f.png"%robustness_weight)

def generate_plot_from_data_intent(robustness_weight):

    ### intent example ###

    xx = np.arange(-2.5, 2, 0.1)
    plt.figure(figsize=(17, 5))
    plt.subplot(1,2,1)
    data = np.load("../data/neural_network/intent/robustness_weight=0.00/data.npy")
    x = data[0,:,:,:]
    y = data[1,:,:,:]
    y_pred = np.load("../data/neural_network/intent/robustness_weight=0.00/output.npy")
    for idx in range(0,1):
        x_history = xx[:10]
        history = x[idx,:,:]
        plt.plot(x_history+2.5, history, c="forestgreen", label="History")
        
        x_future = xx[9:20]
        future = y[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="deepskyblue", label="Future")
        
        x_future = xx[9:]
        future = y_pred[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="coral", label="Prediction")
        
        

    for idx in range(0,10):
        x_history = xx[:10]
        history = x[idx,:,:]
        plt.plot(x_history+2.5, history, c="forestgreen")
        
        x_future = xx[9:20]
        future = y[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="deepskyblue")
        
        x_future = xx[9:]
        future = y_pred[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="coral")

    plt.plot([0,4.5],[0.4, 0.4], 'k--')
    plt.plot([0,4.5],[0.6, 0.6], 'k--')
    plt.xlim([-2.5, 2])
    plt.xlim([-1, 1])
    plt.axis("equal")
    plt.legend(loc="upper left", fontsize=20)
    plt.title("Without robustness regularization", fontsize=26)
    plt.xlabel("Time steps", fontsize=20)
    plt.ylabel("s", fontsize=20)
    plt.tight_layout()
    plt.grid()

    plt.subplot(1,2,2)

    data = np.load("../data/neural_network/intent/robustness_weight=%.2f/data.npy"%robustness_weight)
    x = data[0,:,:,:]
    y = data[1,:,:,:]
    y_pred = np.load("../data/neural_network/intent/robustness_weight=%.2f/output.npy"%robustness_weight)

    for idx in range(0,1):
        x_history = xx[:10]
        history = x[idx,:,:]
        plt.plot(x_history+2.5, history, c="forestgreen", label="History")
        
        x_future = xx[9:20]
        future = y[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="deepskyblue", label="Future")
        
        x_future = xx[9:]
        future = y_pred[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="coral", label="Prediction")
        
        

    for idx in range(0,10):
        x_history = xx[:10]
        history = x[idx,:,:]
        plt.plot(x_history+2.5, history, c="forestgreen")
        
        x_future = xx[9:20]
        future = y[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="deepskyblue")
        
        x_future = xx[9:]
        future = y_pred[idx,:,:]
        future = np.concatenate([history[-1:,:], future], axis=0)
        plt.plot(x_future+2.5, future, c="coral")

    plt.plot([0,4.5],[0.4, 0.4], 'k--')
    plt.plot([0,4.5],[0.6, 0.6], 'k--')
    plt.xlim([-2.5, 2])
    plt.xlim([-1, 1])
    plt.axis("equal")
    plt.legend(loc="upper left", fontsize=20)
    plt.title("With robustness regularization $\gamma=%.2f"%robustness_weight, fontsize=26)
    plt.xlabel("Time steps", fontsize=20)
    plt.ylabel("s", fontsize=20)
    plt.tight_layout()
    plt.grid()


    plt.savefig("../figs/neural_network/intent/intent_comparison_robustness_weight=%.2f.png"%robustness_weight)


if __name__ == '__main__':
    robustness_weight_1 = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    robustness_weight_2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    generate_plot_from_data_bump(robustness_weight_1)
    generate_plot_from_data_intent(robustness_weight_2)