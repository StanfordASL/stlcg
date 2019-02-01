import sys
sys.path.insert(0, '/home/karenleung/repos/trams')
from sumo_interface.sumo_experience_source import *
from sumo_interface.sumo_environment_state_action_features_cartesian import *
import os
from statsmodels.tsa import statespace
from sumo_interface.point import Point
import scene_generator
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict,defaultdict
import copy

# signal_dict = {
# 0 : "nothing",
# 2: "left",
# 8: "brake",
# 10: "left and brake",
# 9: "right and brake",
# 1: "right",
# None: None
# }
signal_dict = defaultdict(lambda: np.NaN)

signal_dict[0] = 0      # nothing
signal_dict[8] = 1      # brake
signal_dict[2] = 2      # left
signal_dict[1] = 3      # right
signal_dict[10] = 4     # left and brake
signal_dict[9] = 5      # right and brake




# fancy model
default_dict = OrderedDict([
    # ego
    # 0
    ('x_ego', lambda m: m['position'][0]),
    ('y_ego', lambda m: m['position'][1]),
    ('theta_ego', lambda m: m['angle']),
    ('omega_ego', lambda m: m['angular_vel']),
    ('v_ego', lambda m: m['speed']),
    ('a_ego', lambda m: m['acceleration']), 
    ('lane_index_ego', lambda m: m['lane_index']),
    ('lane_pos_ego', lambda m: m['lane_pos']), 
    ('signal_ego', lambda m: signal_dict[m['signals']]),
    # front veh
    # 9
    ('x_front_veh', lambda m: m['surrounding_vehs']['front_veh']['position'][0]),
    ('y_front_veh', lambda m: m['surrounding_vehs']['front_veh']['position'][1]),
    ('theta_front_veh', lambda m: m['surrounding_vehs']['front_veh']['angle']),
    ('omega_front_veh', lambda m: m['surrounding_vehs']['front_veh']['angular_vel']),
    ('v_front_veh', lambda m: m['surrounding_vehs']['front_veh']['speed']),
    ('a_front_veh', lambda m: m['surrounding_vehs']['front_veh']['acceleration']), 
    ('lane_index_front_veh', lambda m: m['surrounding_vehs']['front_veh']['lane_index']),
    ('lane_pos_front_veh', lambda m: m['surrounding_vehs']['front_veh']['lane_pos']), 
    ('signal_front_veh', lambda m: signal_dict[m['surrounding_vehs']['front_veh']['signals']]),
    # rear veh
    # 18
    ('x_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['position'][0]),
    ('y_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['position'][1]),
    ('theta_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['angle']),
    ('omega_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['angular_vel']),
    ('v_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['speed']),
    ('a_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['acceleration']), 
    ('lane_index_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['lane_index']),
    ('lane_pos_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['lane_pos']), 
    ('signal_rear_veh', lambda m: signal_dict[m['surrounding_vehs']['rear_veh']['signals']]),
    # rear left veh
    # 27
    ('x_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['position'][0]),
    ('y_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['position'][1]),
    ('theta_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['angle']),
    ('omega_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['angular_vel']),
    ('v_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['speed']),
    ('a_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['acceleration']), 
    ('lane_index_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['lane_index']),
    ('lane_pos_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['lane_pos']), 
    ('signal_rear_left_veh', lambda m: signal_dict[m['surrounding_vehs']['rear_left_veh']['signals']]),        
    # rear right veh
    # 36
    ('x_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['position'][0]),
    ('y_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['position'][1]),
    ('theta_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['angle']),
    ('omega_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['angular_vel']),
    ('v_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['speed']),
    ('a_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['acceleration']), 
    ('lane_index_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['lane_index']),
    ('lane_pos_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['lane_pos']), 
    ('signal_rear_right_veh', lambda m: signal_dict[m['surrounding_vehs']['rear_right_veh']['signals']]),   
    # front left veh
    # 45
    ('x_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['position'][0]),
    ('y_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['position'][1]),
    ('theta_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['angle']),
    ('omega_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['angular_vel']),
    ('v_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['speed']),
    ('a_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['acceleration']), 
    ('lane_index_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['lane_index']),
    ('lane_pos_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['lane_pos']), 
    ('signal_front_left_veh', lambda m: signal_dict[m['surrounding_vehs']['front_left_veh']['signals']]),        
    # front right veh
    # 54
    ('x_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['position'][0]),
    ('y_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['position'][1]),
    ('theta_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['angle']),
    ('omega_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['angular_vel']),
    ('v_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['speed']),
    ('a_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['acceleration']), 
    ('lane_index_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['lane_index']),
    ('lane_pos_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['lane_pos']), 
    ('signal_front_right_veh', lambda m: signal_dict[m['surrounding_vehs']['front_right_veh']['signals']]),   
])


slim_dict = OrderedDict([
    # ego
    # 0
    ('x_ego', lambda m: m['position'][0]),
    ('y_ego', lambda m: m['position'][1]),
    ('theta_ego', lambda m: m['angle']),
    ('v_ego', lambda m: m['speed']),
    # front veh
    # 9
    ('x_front_veh', lambda m: m['surrounding_vehs']['front_veh']['position'][0]),
    ('y_front_veh', lambda m: m['surrounding_vehs']['front_veh']['position'][1]),
    ('theta_front_veh', lambda m: m['surrounding_vehs']['front_veh']['angle']),
    ('v_front_veh', lambda m: m['surrounding_vehs']['front_veh']['speed']),
    # rear veh
    # 18
    ('x_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['position'][0]),
    ('y_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['position'][1]),
    ('theta_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['angle']),
    ('v_rear_veh', lambda m: m['surrounding_vehs']['rear_veh']['speed']),
    # rear left veh
    # 27
    ('x_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['position'][0]),
    ('y_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['position'][1]),
    ('theta_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['angle']),
    ('v_rear_left_veh', lambda m: m['surrounding_vehs']['rear_left_veh']['speed']),
    # rear right veh
    # 36
    ('x_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['position'][0]),
    ('y_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['position'][1]),
    ('theta_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['angle']),
    ('v_rear_right_veh', lambda m: m['surrounding_vehs']['rear_right_veh']['speed']),
    # front left veh
    # 45
    ('x_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['position'][0]),
    ('y_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['position'][1]),
    ('theta_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['angle']),
    ('v_front_left_veh', lambda m: m['surrounding_vehs']['front_left_veh']['speed']),
    # front right veh
    # 54
    ('x_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['position'][0]),
    ('y_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['position'][1]),
    ('theta_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['angle']),
    ('v_front_right_veh', lambda m: m['surrounding_vehs']['front_right_veh']['speed']),
])

def getCollisionDict(collision_filename):
    f  = open(collision_filename, "r")
    collision_dict = defaultdict(list)
    for (i,l) in enumerate(f):
        ll = l.split(' ')
        if len(ll) > 1 and ll[1] == 'Vehicle':
            c1 = ll[2][1:-2]
            c2 = ll[6][1:-2]
            time = float(ll[10].split('=')[-1])
            if not collision_dict[c1]:
                collision_dict[c1] = [c2, time]
    return collision_dict


def getDataFrame(state_trace_adjacent, trial, collision_dict, extract_dict=default_dict, dt = 1.0):
    results = defaultdict(lambda: defaultdict(list))
    t = 0.0

    for (i,sa) in enumerate(state_trace_adjacent):
        for (j, veh_id) in enumerate(sa.vehicle_labels):
            results[veh_id]['t'].append(round(t,3))
            results[veh_id]['trial'].append(trial)
            features = sa.current_vehicle_snapshot[veh_id]
            for (n, extract_func) in extract_dict.items():
                results[veh_id][n].append(extract_func(features))
            if veh_id in collision_dict.keys():
                if t >= collision_dict[veh_id][1]:
                    results[veh_id]['collision'].append(1)
                else:
                    results[veh_id]['collision'].append(0)
            else: 
                results[veh_id]['collision'].append(0)

        t += dt
    return pd.DataFrame(results)


def updateCurrentSnapshot(state, dt=1.0):
    # print(state.previous_vehicle_snapshot.keys(), state.current_vehicle_snapshot.keys(), state.vehicle_labels, '\n')
    for veh_id in state.vehicle_labels:
        if np.abs(state.current_vehicle_snapshot[veh_id]['position'][0]) > 113.50 and state.current_vehicle_snapshot[veh_id]['position'][1] > -6.6:
                state.current_vehicle_snapshot[veh_id]['lane_index'] += 1
        if veh_id in state.current_vehicle_snapshot.keys() and veh_id in state.previous_vehicle_snapshot.keys():

            # Unpack the ego vehicle data
            ego_state = state.current_vehicle_snapshot[veh_id]
            ego_speed = ego_state['speed']
            ego_angle = (ego_state['angle']) % (2*np.pi)

            old_ego_state = state.previous_vehicle_snapshot[veh_id]
            # old_ego_position = old_ego_state['position']
            old_ego_speed = old_ego_state['speed']
            old_ego_angle = (old_ego_state['angle']) % (2*np.pi)
            state.current_vehicle_snapshot[veh_id]['acceleration'] = (ego_speed - old_ego_speed)/dt

            angular_vel = ego_angle - old_ego_angle
            if abs(angular_vel + 2*np.pi) < abs(angular_vel):
                angular_vel = angular_vel + 2*np.pi
            if abs(angular_vel - 2*np.pi) < abs(angular_vel):
                angular_vel = angular_vel - 2*np.pi

            state.current_vehicle_snapshot[veh_id]['angular_vel'] = angular_vel/dt
        elif veh_id in state.current_vehicle_snapshot.keys() and veh_id not in state.previous_vehicle_snapshot.keys(): 
            state.current_vehicle_snapshot[veh_id]['angular_vel'] = 0.0
            state.current_vehicle_snapshot[veh_id]['acceleration'] = 0.0
    return state



def updateSumostateGetAdjacentVeh(sumostate):
    """
    Gets state of other nearby cars during a laneswap manoeuvre. This is dependent on the signal of the ego vehicle (defined by the dictionary below). 
    For example, if the car is not indicating, the only car it cares about it the leader car. If it is indicating left, it will consider the front and 
    rear adjacent cars corresonding to which direction it is indicating. Beware of lanes on the boundary....
        # signal_dict = {
        # 0 : "nothing",
        # 2: "left",
        # 8: "brake",
        # 10: "left and brake",
        # 9: "right and brake",
        # 1: "right"
        # }

    -------------------------------------------
            lane 2
    -------------------------------------------
            lane 1 : exiting cars start here
    -------------------------------------------
            lane 0 : merge lane entering cars start here
    -------------------------------------------
    """
    def _arrangeVehiclesByLane(snapshot):
        d = defaultdict(list)
        for (k,v) in snapshot.items():
            d[v['lane_index']].append(k)
        return d

    def _getAdjacentVehHelper(sumostate, ego_veh_id, relevant_adjacent_lane, lane_veh, ownlane=False):
        vehicle_positions = sumostate.current_vehicle_tree.data
        veh_snapshot = sumostate.current_vehicle_snapshot
        ego_pos = sumostate.current_vehicle_snapshot[ego_veh_id]['position']
        relevant_veh_id = lane_veh[relevant_adjacent_lane].copy()
        if ownlane:
            relevant_veh_id.remove(ego_veh_id)

        # get cars in the relevant adjacent lanes
        relevant_positions = vehicle_positions[[sumostate.vehicle_labels.index(o) for o in relevant_veh_id],:]
        # get relative positions of the ego car
        relative_positions = (relevant_positions - ego_pos)[:,0]

        # get front cars
        front_adjacent_veh = defaultdict(lambda: np.NaN)
        front_adjacent_veh['position'] = [np.NaN, np.NaN]
        relevant_front_veh = relevant_positions[np.where(relative_positions > 0)[0]][:,0]
        if len(relevant_front_veh):
            front_adjacent_veh_id = relevant_veh_id[np.argmax(1/relative_positions)]
            front_adjacent_veh = copy.deepcopy(veh_snapshot[front_adjacent_veh_id])
            if 'surrounding_vehs' in front_adjacent_veh: front_adjacent_veh.pop('surrounding_vehs')
            front_adjacent_veh['id'] = front_adjacent_veh_id

        # get rear cars
        rear_adjacent_veh = defaultdict(lambda: np.NaN)
        rear_adjacent_veh['position'] = [np.NaN, np.NaN]
        relevant_rear_veh = relevant_positions[np.where(relative_positions < 0)[0]][:,0]
        if len(relevant_rear_veh):
            rear_adjacent_veh_id = relevant_veh_id[np.argmax(-1/relative_positions)]
            rear_adjacent_veh = copy.deepcopy(veh_snapshot[rear_adjacent_veh_id])
            if 'surrounding_vehs' in rear_adjacent_veh:  rear_adjacent_veh.pop('surrounding_vehs')
            rear_adjacent_veh['id'] = rear_adjacent_veh_id

        return front_adjacent_veh, rear_adjacent_veh


    lane_veh = _arrangeVehiclesByLane(sumostate.current_vehicle_snapshot)
    
    for (veh_id, state) in sumostate.current_vehicle_snapshot.items():
        
        signal = state['signals']
        lane = state['lane_index']
        ego_pos = state['position']
        sumostate.current_vehicle_snapshot[veh_id]['surrounding_vehs'] = {}

        # get ego lane vehs
        front_veh, rear_veh = _getAdjacentVehHelper(sumostate, veh_id, lane, lane_veh, ownlane=True)
        front_left_veh, rear_left_veh = _getAdjacentVehHelper(sumostate, veh_id, lane+1, lane_veh)
        front_right_veh, rear_right_veh = _getAdjacentVehHelper(sumostate, veh_id, lane-1, lane_veh)

        sumostate.current_vehicle_snapshot[veh_id]['surrounding_vehs']['front_veh'] = front_veh
        sumostate.current_vehicle_snapshot[veh_id]['surrounding_vehs']['rear_veh'] = rear_veh
        sumostate.current_vehicle_snapshot[veh_id]['surrounding_vehs']['rear_left_veh'] = rear_left_veh
        sumostate.current_vehicle_snapshot[veh_id]['surrounding_vehs']['rear_right_veh'] = rear_right_veh
        sumostate.current_vehicle_snapshot[veh_id]['surrounding_vehs']['front_left_veh'] = front_left_veh
        sumostate.current_vehicle_snapshot[veh_id]['surrounding_vehs']['front_right_veh'] = front_right_veh

    return sumostate



def bag_dataframe_to_3d_numpy_helper(data_df_orig, extract_dict):
    long_gap = 6.0
    lat_gap = 2.5
    front_distance = lambda m: np.array(m['x_front_veh']) - np.array(m['x_ego'])
    rear_distance = lambda m: np.array(m['x_ego']) - np.array(m['x_rear_veh'])

    front_left_x = lambda m: np.array(m['x_front_left_veh']) - np.array(m['x_ego'])
    front_right_x = lambda m: np.array(m['x_front_right_veh']) - np.array(m['x_ego'])

    rear_left_x = lambda m: np.array(m['x_ego']) - np.array(m['x_rear_left_veh'])
    rear_right_x = lambda m: np.array(m['x_ego']) - np.array(m['x_rear_right_veh'])

    front_left_y = lambda m: np.array(m['y_front_left_veh']) - np.array(m['y_ego'])
    front_right_y = lambda m: np.array(m['y_ego']) - np.array(m['y_front_right_veh'])

    rear_left_y = lambda m: np.array(m['y_rear_left_veh']) - np.array(m['y_ego'])
    rear_right_y = lambda m: np.array(m['y_ego']) - np.array(m['y_rear_right_veh'])

    data_df = copy.deepcopy(data_df_orig)
    p_max = 0
    trials = data_df.index.levels[0]
    for trial in trials:
        p = len(data_df.loc[trial].keys())
        if p > p_max:
            p_max = p
    M = len(trials) * (p_max - 2)              # number of trials
    N = len(extract_dict.keys())     # number of states
    T = 700                              # max time
    T_max = 0
    array = np.zeros([M, T, N])
    tl = np.zeros((M))
    col = np.zeros((M))
    col_type = np.zeros((M))
    j = 0
    for trial in trials:
        for veh_id, veh_data in data_df.loc[trial].items():
            if veh_id in ['enter.0', 'exit.0'] or 'traffic' in veh_id:
                continue
            collision = 0
            try:
                start_idx = next(x[0] for x in enumerate(veh_data['x_ego']) if x[1] > -113.5)
            except Exception:
                print(veh_id, trial)
                continue
            try:
                end_upper_idx = (next(x[0] for x in enumerate(veh_data['x_ego']) if x[1] > 113.5), -1)
            except Exception:
                end_upper_idx = (len((veh_data['x_ego'])) - 1, -1)
            # check for collisions and get the time.
            front_collision = np.where(front_distance(veh_data) < long_gap)[0]
            rear_collision = np.where(rear_distance(veh_data) < long_gap)[0]

            front_left_collision = np.where(np.logical_and(front_left_x(veh_data) < long_gap, front_left_y(veh_data) < lat_gap) > 0)[0]
            front_right_collision = np.where(np.logical_and(front_right_x(veh_data) < long_gap, front_right_y(veh_data) < lat_gap) > 0)[0] 

            rear_left_collision = np.where(np.logical_and(rear_left_x(veh_data) < long_gap, rear_left_y(veh_data) < lat_gap) > 0)[0]
            rear_right_collision = np.where(np.logical_and(rear_right_x(veh_data) < long_gap, rear_right_y(veh_data) < lat_gap) > 0)[0]            
        
            fc_idx = (len(front_collision) > 0 and front_collision[0]) or -1
            flc_idx = (len(front_left_collision) > 0 and front_left_collision[0]) or -1
            frc_idx = (len(front_right_collision) > 0 and front_right_collision[0]) or -1
            rc_idx = (len(rear_collision) > 0 and rear_collision[0]) or -1
            rlc_idx = (len(rear_left_collision) > 0 and rear_left_collision[0]) or -1
            rrc_idx = (len(rear_right_collision) > 0 and rear_right_collision[0]) or -1

            b = []
            if fc_idx > start_idx:
                b.append(fc_idx)

            if fc_idx > start_idx or flc_idx > start_idx or frc_idx > start_idx or rc_idx > start_idx or rlc_idx > start_idx or rrc_idx > start_idx:
                b = [((fc_idx > start_idx)*fc_idx, 0), ((flc_idx > start_idx)*flc_idx, 1), ((frc_idx > start_idx)*frc_idx, 2), ((rc_idx > start_idx)*rc_idx, 3), ((rlc_idx > start_idx)*rlc_idx, 4), ((rrc_idx > start_idx)*rrc_idx, 5)]
                bb = [(i,bi) for (i,bi) in b if i > 0]
                end_collision_idx = min(bb)
            else:
                end_collision_idx = end_upper_idx
                

            center_lanes = [-8.25, -4.95, -1.65,]
            start_lane = veh_data['lane_index_ego'][0]
            target_lane = (start_lane + 1) % 2
            end_lc_idx = np.where(np.abs(np.array(veh_data['y_ego']) - center_lanes[target_lane]) < 0.8)[0]
            if len(end_lc_idx) > 0:
                end_lc_idx = (end_lc_idx[0] + 10, -1)
            else:
                end_lc_idx = end_upper_idx
            end_idx = min([end_upper_idx, end_lc_idx, end_collision_idx])
            if end_collision_idx[0] <= end_idx[0]:
                collision = 1

            t_max = end_idx[0] - start_idx + 1
            if t_max > T_max:
                T_max = t_max
            for (i,s) in enumerate(extract_dict.keys()):
                tl[j] = t_max
                col[j] = collision
                col_type[j] = end_idx[1]
                if s not in  ['trial', 't', 'collision']:
                    # print(t_max, len(veh_data[s][start_idx:end_idx+1]), start_idx, end_idx, len(veh_data[s]), veh_id, trial)
                    array[j, :t_max, i] = veh_data[s][start_idx:end_idx[0]+1]
            j += 1

    return array[:j, :T_max,:], tl[:j], col[:j], col_type[:j]
                

def bag_dataframe_to_3d_numpy(data_df_orig, extract_dict):
    data_np, tl, col, col_type = bag_dataframe_to_3d_numpy_helper(data_df_orig, extract_dict)
    data = {k:data_np[:,:,i] for (i,k) in enumerate(extract_dict.keys())}
    cars = ['front_veh', 'front_left_veh', 'front_right_veh', 'rear_veh', 'rear_left_veh', 'rear_right_veh']
    
    def _fill_nan_values(data, car):
        long_sign = 1
        lat_sign = 1
        if 'rear' in car:
            long_sign = -1
        elif 'front' in car:
            long_sign = 1

        if 'right' in car:
            lat_sign = -1
        elif 'left' in car:
            lat_sign = 1
        else:
            lat_sign = 0

        data['x_'+car][np.isnan(data['x_'+car])] = data['x_ego'][np.isnan(data['x_'+car])] + long_sign * 50
        data['y_'+car][np.isnan(data['y_'+car])] = data['y_ego'][np.isnan(data['y_'+car])] + lat_sign * 3.3
        data['theta_'+car][np.isnan(data['theta_'+car])] = np.pi/2
        # data['omega_'+car][np.isnan(data['omega_'+car])] = 0.0
        data['v_'+car][np.isnan(data['v_'+car])] = data['v_ego'][np.isnan(data['v_'+car])]
        # data['a_'+car][np.isnan(data['a_'+car])] = data['a_ego'][np.isnan(data['a_'+car])]
        # data['lane_index_'+car][np.isnan(data['lane_index_'+car])] = data['lane_index_ego'][np.isnan(data['lane_index_'+car])] + lat_sign * 1
        # data['lane_pos_'+car][np.isnan(data['lane_pos_'+car])] = data['lane_pos_ego'][np.isnan(data['lane_pos_'+car])] + long_sign * 50
        # data['signal_'+car][np.isnan(data['signal_'+car])] = 0

    _fill_nan_values(data, 'front_veh')
    _fill_nan_values(data, 'rear_veh')
    _fill_nan_values(data, 'front_left_veh')
    _fill_nan_values(data, 'front_right_veh')
    _fill_nan_values(data, 'rear_left_veh')
    _fill_nan_values(data, 'rear_right_veh')

    return data_np, tl, col, col_type

def frontCarCrossOverTime(veh_data, start_end_idx):
    print(start_end_idx)
    start_idx, end_idx = start_end_idx
    dx = np.array(veh_data['front_veh_distance_ego'])[start_idx:end_idx+1]
    dv = np.array(veh_data['v_front'])[start_idx:end_idx+1] - np.array(veh_data['v_ego'])[start_idx:end_idx+1]
    return -dx/dv