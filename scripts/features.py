import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

params = {'a_min': -6,
		  't_reaction': 0.3,
		  'longitudinal_min': 5.5,
		  'lateral_min': 2.25,
		  'speed_limit': 16

		 }


def front_stopping_distance(trace, params, key_idx):
	'''
	This quantity describes the stopping distance 
	between the ego vehicle and the front, front left and front right vehicles.
	If both vehicles were to slam on the brakes, and the ego vehicle 
	has some reaction time (t_reaction), calculate the 
	time to collision.
	The ego car should be able to stop behind the front 
	car with some minimum gap d_min. The maximum time to collision is capped at ttc_max
	
	We want the vehicles to be "safe".
	(Format ready to be used for STL)

	'''     
	def _fsd(x1, x2, v1, v2, tr, a, xmin):
		dx = x1 - x2
		ttc_max = -v1/a
		t = -v1 / a
		x1_stop = dx + v1*t + 0.5*a * t**2
		t = (a*tr - v2) / a
		x2_stop = v2* t + 0.5 * a * (t - tr)**2
		return x1_stop - x2_stop
		# ttc_short = [(-(v1-v2) - np.sqrt((v1-v2)**2 - 2*a*(dx - xmin)))/a, (-(v1-v2) + np.sqrt((v1-v2)**2 - 2*a*(dx - xmin)))/a]
		# try:
		# 	ttc_short = min([ttc for ttc in ttc_short if ttc > 0])
		# except:
		# 	ttc_short = 1
		# ttc_long = (xmin - dx + 0.5 * a * tr)/(v1 - v2 + 0.5 * a * tr)
		# ttc_long = ttc_max if ttc_long < 0 else ttc_long
		# return min(ttc_short if ttc_short < tr else ttc_long, ttc_max)


	a = params['a_min']
	tr = params['t_reaction']
	xmin = params['longitudinal_min']

	V2 = trace[:, key_idx['v_ego']]
	V1 = trace[:, key_idx['v_front_veh']]
	X2 = trace[:, key_idx['x_ego']]
	X1 = trace[:, key_idx['x_front_veh']]
	ttc_front = [_fsd(x1, x2, v1, v2, tr, a, xmin) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]

	V1 = trace[:, key_idx['v_front_left_veh']]
	X1 = trace[:, key_idx['x_front_left_veh']]
	ttc_front_left = [_fsd(x1, x2, v1, v2, tr, a, xmin) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]

	V1 = trace[:, key_idx['v_front_right_veh']]
	X1 = trace[:, key_idx['x_front_right_veh']]
	ttc_front_right = [_fsd(x1, x2, v1, v2, tr, a, xmin) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]

	return ttc_front, ttc_front_left, ttc_front_right


def crossover_time_rear(trace, key_idx, rear=True):
	'''
	This quantity describes the crossover distance
	between the two cars (ego (v2) and the front car (v1)).
	If both cars were to slam on the brakes, and the ego car 
	has some reaction time (t_reaction), calculate the 
	stopping distance of both cars.
	The ego car should be able to stop behind the front 
	car with some minimum gap d_min.
	
	We want 
	-(x1 - x2 + 0.5/np.abs(a_min) * (v1**2 - v2**2) - v2 * t_reaction - front_min) < 0 
	for the vehicles to be "safe".
	(Format ready to be used for STL)

	'''     
	def _cot(x1, x2, v1, v2):
		dx = x1 - x2
		dv = v1 - v2
		if dv == 0 :
			dv = 0.01
		dv = np.sign(dv)*max(np.abs(dv), 0.01)
		return max(min(-dx / dv, 10), -10)
		
	X2 = trace[:, key_idx['x_ego']]
	V2 = trace[:, key_idx['v_ego']]

	X1 = trace[:, key_idx['x_front_veh']]
	V1 = trace[:, key_idx['v_front_veh']]
	cot_front = [_cot(x1, x2, v1, v2) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]
	
	if rear: 
		V1 = trace[:, key_idx['v_rear_left_veh']]
		X1 = trace[:, key_idx['x_rear_left_veh']]
		cot_rear_left = [_cot(x1, x2, v1, v2) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]

		V1 = trace[:, key_idx['v_rear_right_veh']]
		X1 = trace[:, key_idx['x_rear_right_veh']]
		cot_rear_right = [_cot(x1, x2, v1, v2) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]

		return cot_front, cot_rear_left, cot_rear_right
	else:
		V1 = trace[:, key_idx['v_front_left_veh']]
		X1 = trace[:, key_idx['x_front_left_veh']]
		cot_front_left = [_cot(x1, x2, v1, v2) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]

		V1 = trace[:, key_idx['v_front_right_veh']]
		X1 = trace[:, key_idx['x_front_right_veh']]
		cot_front_right = [_cot(x1, x2, v1, v2) for (x1, x2, v1, v2) in zip(X1, X2, V1, V2)]

		return cot_front, cot_front_left, cot_front_right

def front_adjacent_car_gap(trace, params, key_idx):
	'''
	This quantity describes how close you are to your adjacent vehicles.
	If the lateral distance is less than lateral_min and if the longitudinal distance is less than longitudinal_min,
	then the cars are considered to be in collision.

	'''
	x_ego = trace[:, key_idx['x_ego']].copy()
	y_ego = trace[:, key_idx['y_ego']].copy()

	x_front_veh = trace[:, key_idx['x_front_veh']].copy()
	y_front_veh = trace[:, key_idx['y_front_veh']].copy()


	x_front_left_veh = trace[:, key_idx['x_front_left_veh']].copy()
	y_front_left_veh = trace[:, key_idx['y_front_left_veh']].copy()

	x_front_right_veh = trace[:, key_idx['x_front_right_veh']].copy()
	y_front_right_veh = trace[:, key_idx['y_front_right_veh']].copy()

	dx_front = np.abs(x_front_veh - x_ego)
	dy_front = np.abs(y_front_veh - y_ego)

	dx_front_left = np.abs(x_front_left_veh - x_ego)
	dy_front_left = np.abs(y_front_left_veh - y_ego)

	dx_front_right = np.abs(x_front_right_veh - x_ego)
	dy_front_right = np.abs(y_front_right_veh - y_ego)

	return dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right



def rear_adjacent_car_gap(trace, params, key_idx):
	'''
	This quantity describes how close you are to your adjacent vehicles.
	If the lateral distance is less than lateral_min and if the longitudinal distance is less than longitudinal_min,
	then the cars are considered to be in collision.

	'''
	x_ego = trace[:, key_idx['x_ego']].copy()
	y_ego = trace[:, key_idx['y_ego']].copy()

	x_rear_veh = trace[:, key_idx['x_rear_veh']].copy()
	y_rear_veh = trace[:, key_idx['y_rear_veh']].copy()


	x_rear_left_veh = trace[:, key_idx['x_rear_left_veh']].copy()
	y_rear_left_veh = trace[:, key_idx['y_rear_left_veh']].copy()

	x_rear_right_veh = trace[:, key_idx['x_rear_right_veh']].copy()
	y_rear_right_veh = trace[:, key_idx['y_rear_right_veh']].copy()

	dx_rear = np.abs(x_rear_veh - x_ego)
	dy_rear = np.abs(y_rear_veh - y_ego)

	dx_rear_left = np.abs(x_rear_left_veh - x_ego)
	dy_rear_left = np.abs(y_rear_left_veh - y_ego)

	dx_rear_right = np.abs(x_rear_right_veh - x_ego)
	dy_rear_right = np.abs(y_rear_right_veh - y_ego)

	return dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right



def nothing(trace, key_idx):
	return 1*(trace[:,key_idx['signal_ego']] == 0)

def braking(trace, key_idx):
	return 1*(trace[:,key_idx['signal_ego']] == 1)

def indicating_left(trace, key_idx):
	return 1*(trace[:,key_idx['signal_ego']] == 2)

def indicating_left_brake(trace, key_idx):
	return 1*(trace[:,key_idx['signal_ego']] == 4)

def indicating_right(trace, key_idx):
	return 1*(trace[:,key_idx['signal_ego']] == 3)

def indicating_right_brake(trace, key_idx):
	return 1*(trace[:,key_idx['signal_ego']] == 5)

def indicating(trace, key_idx):
	return (nothing(trace, key_idx),
		   braking(trace, key_idx),
		   indicating_left(trace, key_idx), 
		   indicating_left_brake(trace, key_idx), 
		   indicating_right(trace, key_idx), 
		   indicating_right_brake(trace, key_idx))


def depart_lane(trace, key_idx):
	start_lane = trace[0, key_idx['lane_index_ego']]
	return trace[:,key_idx['lane_index_ego']] - start_lane

def lane_change_direction(trace, key_idx):
	lane_idx = trace[0,key_idx['lane_index_ego']]
	return 'left' if lane_idx == 0 else 'right'