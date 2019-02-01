import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from abc import ABC, abstractmethod
import sympy as sym
# lane plotting


def plotLanes():
    l2 = -1.65
    l1 = -4.95
    l0 = -8.25
    lane_width = np.abs((l2 - l0)/2)
    ys = -np.array([lane_width * i for i in range(4)])
    ys = np.stack([ys, ys], axis=0)
    plt.plot(np.concatenate([-np.ones([1,4]), np.ones([1,4])], axis=0)*150, ys)


def plotTrial(i, j, data_df):
    cols = [(1,0,0), (0,1,0), (0,0,1)]
    typ = ["enter", "exit", "traffic"]
    trial = 'trial_' + str(j)
    l = 2.8
    b = 1.1
    fig, ax = plt.subplots(figsize=(70, 3))
    t = round(i/10.0, 1)
    collision_flag = False
    patch_collection = []
    color = []
    for (veh_id, state) in data_df.loc[trial].items():
        if t < state['t'][0] or t > state['t'][-1]:
            continue
        else:
            idx = int((t - state['t'][0])*10)
            col = cols[typ.index(veh_id.split('.')[0])]
            theta = state['theta_ego'][idx]*180/np.pi
            phi = 90 - theta
            theta = state['theta_ego'][idx]
            pos = (state['x_ego'][idx], state['y_ego'][idx])
            xy = (pos[0] - l*np.sin(theta) + b*np.cos(theta), pos[1] - l*np.cos(theta) - b*np.sin(theta))
            color.append(col)
            rectangle = Rectangle(xy, 2*l, 2*b, color=col, angle=phi)
            patch_collection.append(rectangle)
            ax.text(pos[0]-1.5, pos[1]-1.55, veh_id, fontsize=12)
            collision_begin = -1
            
            if state['collision'][idx] == 1:
                ax.scatter(pos[0], pos[1], s=10000, facecolors='none', edgecolors='r')
                collision_flag = True
        if collision_flag:
            ax.text(-149, -1.25, 'time: '+ str(t), fontsize=18, color='red')
        else:
            ax.text(-149, -1.25, 'time: '+ str(t), fontsize=18)
            
    pc = PatchCollection(patch_collection, facecolor=color)
    ax.add_collection(pc)
    plotLanes()
    plt.axis([-150, 150, -10, 0])
    plt.show()         

def get_start_end_index(veh_data):
    fvd = np.array(veh_data['front_veh_distance_ego'])
    f = np.where(fvd > -1)[0]
    if len(f) < 10:
        return None
    difff = np.diff(f) 
    
    bp = np.where(difff > 1)[0]
    bp = np.concatenate([[0], bp, [len(difff)]])
    argmax = np.argmax(np.diff(bp))
    start_idx = f[bp[argmax] + 1] - 1*(len(bp)==3)
    end_idx = f[bp[argmax+1]]
    
    return (start_idx, end_idx)

def plotRelevantSection(data_df, features):

    def plotRelevantSectionVehicle(data_df, trial, features, veh_id):
        if type(trial) == int:
            trial = 'trial_' + str(trial)
            data = data_df.loc[trial]

        veh_data = data[veh_id]
        # start_end_idx = get_start_end_index(veh_data)
        fn = len(features)
        # if start_end_idx:
        plt.figure(figsize=(70, 5*fn))
        for (i,fns) in enumerate(features):
            x = fns[0]
            f = fns[1]
            name = fns[2]
            plt.subplot(fn, 1, i+1)
            plt.scatter(x(veh_data), f(veh_data))
            if 'position' in name:
                plotLanes()
            plt.title(name)
            plt.xlim([-150, 150])
            # plt.plot([0, 60], [0, 0])
            try:
                plt.ylim(fns[3])
            except Exception:
                pass
        plt.show()

    N = len(data_df.index.levels[0])
    interact(plotRelevantSectionVehicle, data_df=fixed(data_df), trial=(0, N-1), features=fixed(features), veh_id=(data_df.loc['trial_0'].keys()))

def convertArrayToEpisode(array_list, var_name):
    
    for (i,vn) in enumerate(var_name):
        if i == 0:
            symbols = var_name[0]
        else:
            symbols += ' ' + vn
    syms = sym.symbols(symbols)
    if type(syms) is not tuple:
        syms = [syms]
    episode = []

    for i in range(len(array_list[0])):
        entry = {}
        for (j,v) in enumerate(syms):
            entry[v] = array_list[j][i]
        episode.append(entry)
    return syms, episode

def plotStateTrace(data_np, tl, col):
    def foo(i, k, data_np, tl, col):
        plt.figure(figsize=(20, 3))
        j = int(min(k, tl[i]-1))
        plt.scatter(data_np[i,j, 0], data_np[i,j, 1], c='r', marker='*', s=100)
        plt.scatter(data_np[i,j, 45], data_np[i,j, 46])
        plt.scatter(data_np[i,j, 54], data_np[i,j, 55])
        plt.scatter(data_np[i,j, 9], data_np[i,j, 10])
        plt.scatter(data_np[i,j, 27], data_np[i,j, 28])
        plt.scatter(data_np[i,j, 36], data_np[i,j, 37])
        plt.scatter(data_np[i,j, 18], data_np[i,j, 19])
        plotLanes()
        plt.xlim([-113, 113])
        plt.show()
        
    interact(foo, i=(0, int(len(tl))), k=(0,int(max(tl))), data_np=fixed(data_np), tl=fixed(tl), col=fixed(col))