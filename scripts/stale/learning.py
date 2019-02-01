import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tensorflow as tf

# import stl.stl as stl
import IPython
import sympy as sym
# from stl_main import *
# from data_collection import *
# from features import *
import utils




import sys
sys.path.insert(0, '/home/karenleung/repos/trams')
import os

import stl.stl as stl

# ========================================================================================================================================
# STL RNN Approximation

class Maxish(torch.nn.Module):
    def __init__(self):
        super(Maxish, self).__init__()

    def forward(x, dim=1, scale=10):
        if scale >= 0:
            if dim == 0:
                return torch.sum(torch.exp(scale*x)/torch.sum(torch.exp(scale*x), dim=dim)*x).unsqueeze(dim)
            else:
                return torch.sum(torch.exp(scale*x)/torch.sum(torch.exp(scale*x), dim=dim).unsqueeze(-1)*x, dim=dim).unsqueeze(-1)
        else:
            return torch.max(x, dim=dim, keepdim=True)[0]

class Minish(torch.nn.Module):
    def __init__(self):
        super(Minish, self).__init__()

    def forward(x, dim=1, scale=10):
        if scale >= 0:
            if dim == 0:
                return torch.sum(torch.exp(-scale*x)/torch.sum(torch.exp(-scale*x), dim=dim)*x).unsqueeze(dim)
            else:
                return torch.sum(torch.exp(-scale*x)/torch.sum(torch.exp(-scale*x), dim=dim).unsqueeze(-1)*x, dim=dim).unsqueeze(-1)
        else:
            return torch.min(x, dim=dim, keepdim=True)[0]

class TemporalOperator(torch.nn.Module):
    def __init__(self, interval=None):
        super(TemporalOperator, self).__init__()
        self.interval = interval
        self._interval = stl.Interval(0, np.inf) if self.interval is None else self.interval
        self.rnn_dim = 1 if not self.interval else self.interval.upper()
        self.steps = 1 if not self.interval else self.interval.upper() - self.interval.lower() + 1
    
    def robustness(self, x, dim=1, scale=5, cuda=False):
        outputs, states =  self.run_cell(x, self.initialize_rnn_cell(x, dim=dim, cuda=cuda), dim=dim, scale=scale)
        return outputs[-1]

        
    def _initialize_rnn_cell(self, x, dim, init_val, cuda=False):
        rnn_dim = self.rnn_dim

        if dim == 0:
            h0 = torch.tensor(np.ones([rnn_dim])*init_val)
        else:
            batch_size = x.shape[0]
            h0 = torch.tensor(np.ones([batch_size, rnn_dim])*init_val)

        return h0.cuda() if cuda else h0
    
    def initialize_rnn_cell(self, x, dim=1, cuda=False):
        raise NotImplementedError("initialize_rnn_cell not implemented yet")
    
    def rnn_cell(self, x, h0, dim=1, scale=5):
        if self.operation is None:
            raise Exception()

        if self.interval is None:
            input_ = torch.cat([h0, x], dim=dim)
            output = state = self.operation(input_, dim=dim, scale=scale)
        else:
            if dim == 0:
                input_ = torch.cat([h0, x], dim=dim)[:self.steps]
                output = self.operation(input_, dim=dim, scale=scale)
                state = torch.cat([h0[1:], x], dim=dim)
            elif dim == 1:
                input_ = torch.cat([h0, x], dim=dim)[:,:self.steps]
                output = self.operation(input_, dim=dim, scale=scale)
                state = torch.cat([h0[:,1:], x], dim=dim)
            else:
                raise NotImplementedError("dim = ", dim, " is not implemented")
        return output, state

    def run_cell(self, x, h0, dim=1, scale=5):
        outputs = []
        states = []
        h = h0
        xs = torch.split(x, 1, dim=dim)
        for i in range(len(xs)):
            o, h = self.rnn_cell(xs[i], h, dim=dim, scale=scale)
            outputs.append(o)
            states.append(h)

        return outputs, states

    def forward(self, x, dim=1, scale=5):
        raise NotImplementedError()
    
    def __str__(self):
        raise NotImplementedError()
    
class Always(TemporalOperator):
    def __init__(self, interval=None, comp=False):
        super(Always, self).__init__(interval)
        self.operation = Minish.forward
        self.comp = comp
        
    def initialize_rnn_cell(self, x, dim=1, cuda=False):
        return self._initialize_rnn_cell(x, dim, 1E2, cuda=cuda)
    

    def forward(self, x, dim=1, scale=5, cuda=False):
        if self.interval:
            if self.interval.upper() - self.interval.lower() >= x.shape[-1]:
                raise ValueError("Interval is longer than the trace, except the infinite case.")
        if self.interval is not None and self.comp is False:
            if dim == 0:
                # x = x[self.interval.lower():self.interval.upper()+1]
                x = x[-self.interval.upper()-1:]
            elif dim == 1:
                # x = x[:,self.interval.lower():self.interval.upper()+1]
                x = x[:,-self.interval.upper()-1:]
            else:
                raise NotImplementedError("dim = ", dim, " is not implemented")
        return self.robustness(x, dim=dim, scale=scale, cuda=cuda)
    
    def __str__(self):
        return "◻ " + str(self._interval)
        
class Eventually(TemporalOperator):
    def __init__(self, interval=None, comp=False):
        super(Eventually, self).__init__(interval)
        self.operation = Maxish.forward
        self.comp = comp
        
    def initialize_rnn_cell(self, x, dim=1, cuda=False):
        return self._initialize_rnn_cell(x, dim, -1E2, cuda=cuda)
    
    def forward(self, x, dim=1, scale=5, cuda=False):
        if self.interval:
            if self.interval.upper() - self.interval.lower() >= x.shape[-1]:
                raise ValueError("Interval is longer than the trace, except the infinite case.")
        if self.interval is not None and self.comp is False:
            if dim == 0:
                # x = x[self.interval.lower():self.interval.upper()+1]
                x = x[-self.interval.upper()-1:]
            elif dim == 1:
                # x = x[:,self.interval.lower():self.interval.upper()+1]
                x = x[:,-self.interval.upper()-1:]
            else:
                raise NotImplementedError("dim = ", dim, " is not implemented")
        return self.robustness(x, dim=1, scale=scale, cuda=cuda)
    
    def __str__(self):
        return "◇ " + str(self._interval)

class TemporalComposition(torch.nn.Module):
    def __init__(self, inner, outer):
        super(TemporalComposition, self).__init__()
        self.outer = outer
        self.outer.comp = True
        self.inner = inner
        self.inner.comp = True
        self.interval = None

    def run(self, x, dim=1, scale=5, cuda=False):
        # if self.outer.interval is not None:
        #     if dim == 0:
        #         x = x[self.outer.interval.lower():self.outer.interval.upper()+1]
        #     elif dim == 1:
        #         x = x[:,self.outer.interval.lower():self.outer.interval.upper()+1]
        #     else:
        #         raise NotImplementedError("dim = ", dim, " is not implemented")
        inner_outputs, inner_states = self.inner.run_cell(x, self.inner.initialize_rnn_cell(x, cuda=cuda), dim=dim, scale=scale)
        s = 0 if self.inner.interval is None else self.inner.interval.upper()
        new_x = torch.cat(inner_outputs[s:], dim=dim)
        outer_outputs, outer_states = self.outer.run_cell(new_x, self.outer.initialize_rnn_cell(new_x, cuda=cuda), dim=dim, scale=scale)
        return (outer_outputs, outer_states), (inner_outputs, inner_states)
    
    def forward(self, x, dim=1, scale=5, cuda=False):
        (outer_outputs, outer_states), (inner_outputs, inner_states) = self.run(x, dim=dim, scale=scale, cuda=cuda)
        return outer_outputs[-1]
    
    def __str__(self):
        return str(self.outer) + str(self.inner) 



class LessThan(torch.nn.Module):
    def __init__(self, lhs="x", rhs = 0.0):
        super(LessThan, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        
    def forward(self, x, y=None, scale=5, exaggeration_factor=1):
        return (self.rhs - x)*exaggeration_factor
    
    def __str__(self):
        return self.lhs + " <= " + str(self.rhs)
    
class GreaterThan(torch.nn.Module):
    def __init__(self, lhs="x", rhs = 0.0):
        super(GreaterThan, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        
    def forward(self, x, y=None, scale=5, exaggeration_factor=1):
        return (x - self.rhs)*exaggeration_factor
    
    def __str__(self):
        return self.lhs + " >= " + str(self.rhs)

class Equal(torch.nn.Module):
    def __init__(self, lhs="x", rhs = 0.0, exaggeration_factor=1):
        super(Equal, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        
    def forward(self, x, y=None, scale=5):
        return torch.abs(x - self.rhs)*exaggeration_factor
    
    def __str__(self):
        return self.lhs + " == " + str(self.rhs)
    

        
class Negation(torch.nn.Module):
    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula
        
    def forward(self, x, y=None, scale=5, exaggeration_factor=1):
        return -self.subformula(x)*exaggeration_factor
    
    def __str__(self):
        return "¬(" + str(self.subformula) + ")"
    
class And(torch.nn.Module):
    def __init__(self, subformula1, subformula2):
        super(And, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        
    def forward(self, x, y, scale=5):
        a = self.subformula1(x, scale=scale)
        b = self.subformula2(y, scale=scale)
        ab = torch.stack([a, b], dim=2)
        return Minish.forward(ab, dim=2, scale=scale).squeeze(-1)
    
    def __str__(self):
        return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"
    
class Or(torch.nn.Module):
    def __init__(self, subformula1, subformula2):
        super(Or, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        
    def forward(self, x, y, scale=5):
        a = self.subformula1(x, scale=scale)
        b = self.subformula2(y, scale=scale)
        ab = torch.stack([a, b], dim=2)
        return Maxish.forward(ab, dim=2, scale=scale).squeeze(-1)
    
    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"
    

class STLModel(torch.nn.Module):
    def __init__(self, inner, outer):
        super(STLModel, self).__init__()
        self.inner = inner
        self.outer = outer
    def __str__(self):
        return str(self.outer) + " ( " + str(self.inner) + " ) "
    
    def setup(self, x, y=None, scale=5):
        return self.inner(x, y, scale=5)
    
    def forward(self, x, y=None, dim=1, scale=5, cuda=False):
        if self.outer.interval is not None:
            if dim == 0:
                x = x[self.outer.interval.lower():self.outer.interval.upper()+1]
                if y is not None:
                    y = y[self.outer.interval.lower():self.outer.interval.upper()+1]
            elif dim == 1:
                x = x[:,self.outer.interval.lower():self.outer.interval.upper()+1]
                if y is not None:
                    y = y[:,self.outer.interval.lower():self.outer.interval.upper()+1]
            else:
                raise NotImplementedError("dim = ", dim, " is not implemented")
        x_input = self.setup(x, y, scale=scale)
        return self.outer(x_input, dim=dim, scale=scale, cuda=cuda)

def sigmoid(x):
    return 1/(1+np.exp(-x))



def get_τ(x_std, A, scale=100000, device=torch.device('cuda'), cuda=True):
    τ_max = max(5, int(x_std.shape[-1]/2))
    τ_list = range(3, τ_max)
    best_loss = torch.tensor([np.inf], dtype=torch.float64, device=device)
    best_τ = τ_list[0]
    for (i,τ) in enumerate(τ_list):
        tin = Always(stl.Interval(0, τ))
        tout = Eventually()
        outer = TemporalComposition(tin, tout)
        f = STLModel(A, outer)
        loss = torch.abs(f(x_std, scale=scale, cuda=cuda))
        if loss <= best_loss:
            best_loss = loss
            best_τ = τ
        del tin
        del tout
        del outer
        del f
    return best_τ-1, best_loss

def grid_sample(τ, x_std, y_std, A, β_min=-4, β_max=4, scale=100000, device=torch.device('cuda'), cuda=True):
    β_list = torch.arange(-3, 3, 0.1, dtype=torch.float64, device=device)
    best_loss = torch.tensor([np.inf], dtype=torch.float64, device=device)
    best_β = β_list[0]
    for (i,β) in enumerate(β_list):
        B = GreaterThan("fsd", β)    
        inner = And(A, B)
        tin = Always(stl.Interval(0, τ))
        tout = Eventually()
        outer = TemporalComposition(tin, tout)
        formula = STLModel(inner, outer)
        loss = torch.abs(formula(x_std, y_std, scale=scale, cuda=cuda))
        if loss <= best_loss:
            best_loss = loss
            best_β = β
        del inner
        del tin
        del tout
        del outer
        del formula
    return best_β, best_loss

def scale_schedule(epoch, NUM_EPOCH):
    return float(20*sigmoid((epoch - 1*NUM_EPOCH/10)/50))

def front_quantities(save_filename):
    data = np.load("/home/karenleung/Documents/simulator/lanechange/data/npz/lanechange_fancy.npz")
    data_np = data["data_np"]
    tl = data["tl"]
    key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
    N_data = len(tl)
    ξs = []
    αs = []
    for idx in range(N_data):
        print("--------------------------- Index ", idx, "---------------------------")
        tl_ = int(tl[idx])
        car = data_np[idx,:tl_,:]
        key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
        # distance between front vehicles
        (dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right) = front_adjacent_car_gap(car, params, key_idx)
        # distance between rear vehicles
        (dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right) = rear_adjacent_car_gap(car, params, key_idx)
        # stopping distances between front vehiclesii
        (fsdf, fsdfl, fsdfr) = front_stopping_distance(car, params, key_idx)
        # crossover time between rear adjacent and fron vehicles
        (cotf, cotrl, cotrr) = crossover_time(car, key_idx)


        data_list = [dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right,
                     dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right,
                     fsdf, fsdfl, fsdfr,
                     cotf, cotrl, cotrr,
                     car[:,key_idx['lane_index_ego']]]
        var_name = ['dxf', 'dyf', 'dxfl', 'dyfl', 'dxfr', 'dyfr',
                    'dxr', 'dyr', 'dxrl', 'dyrl', 'dxrr', 'dyrr',
                    'fsdf', 'fsdfl', 'fsdfr',
                    'cotf', 'cotrl', 'cotrr',
                    'lane_idx']

        (dxf, dyf, dxfl, dyfl, dxfr, dyfr, dxr, dyr, dxrl, dyrl, dxrr, dyrr, fsdf, fsdfl, fsdfr, cotf, cotrl, cotrr, li), episode = utils.convertArrayToEpisode(data_list, var_name)
        trace = stl.Trace(episode)
        αs.append(min(trace.series[fsdf]))
        ξs.append(min(trace.series[cotf]))
    np_dict = {'αs': αs, 'ξs': ξs}
    np.savez(save_filename, **np_dict)


def get_τs(save_filename, scale=100, device = torch.device('cuda'), cuda=True):
    data = np.load("/home/karenleung/Documents/simulator/lanechange/data/npz/lanechange_fancy.npz")
    data_np = data["data_np"]
    tl = data["tl"]
    key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
    N_data = len(tl)
    τs = []
    features_stats = np.load("/home/karenleung/Documents/simulator/lanechange/scripts/stl_data/features_stats.npz")
    dy_μ, dy_σ = float(features_stats['dy_μ']), float(features_stats['dy_σ'])
    dy_thres = torch.tensor(3.5, dtype=torch.float64, device=device)
    A = LessThan("dy", ((dy_thres - dy_μ)/dy_σ))
    for idx in range(N_data):
        print("--------------------------- Index ", idx, "---------------------------")
        tl_ = int(tl[idx])
        car = data_np[idx,:tl_,:]
        key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
        # distance between front vehicles
        (dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right) = front_adjacent_car_gap(car, params, key_idx)
        # distance between rear vehicles
        (dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right) = rear_adjacent_car_gap(car, params, key_idx)
        # stopping distances between front vehiclesii
        (fsdf, fsdfl, fsdfr) = front_stopping_distance(car, params, key_idx)
        # crossover time between rear adjacent and fron vehicles
        (cotf, cotrl, cotrr) = crossover_time(car, key_idx)


        data_list = [dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right,
                     dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right,
                     fsdf, fsdfl, fsdfr,
                     cotf, cotrl, cotrr,
                     car[:,key_idx['lane_index_ego']]]
        var_name = ['dxf', 'dyf', 'dxfl', 'dyfl', 'dxfr', 'dyfr',
                    'dxr', 'dyr', 'dxrl', 'dyrl', 'dxrr', 'dyrr',
                    'fsdf', 'fsdfl', 'fsdfr',
                    'cotf', 'cotrl', 'cotrr',
                    'lane_idx']

        (dxf, dyf, dxfl, dyfl, dxfr, dyfr, dxr, dyr, dxrl, dyrl, dxrr, dyrr, fsdf, fsdfl, fsdfr, cotf, cotrl, cotrr, li), episode = utils.convertArrayToEpisode(data_list, var_name)
        trace = stl.Trace(episode)
        lcd = lane_change_direction(car, key_idx)
        if lcd == "left":
            dy = dyfl
            fsd = fsdfl
            cot = cotrl
        else:
            dy = dyfr
            fsd = fsdfr
            cot = cotrr

        dy_trace_ = [trace.series[dy][-k-1]  for k in range(len(trace.series[dy]))]
        dy_trace = torch.tensor(dy_trace_, dtype=torch.float64, device=device).unsqueeze(0)
        dy_trace_std = (dy_trace - dy_μ)/dy_σ

        best_τ, _ = get_τ(dy_trace_std, A, scale=100)
        print("Best τ: ", best_τ)
        τs.append(best_τ)
    np_dict = {'τs': τs}
    np.savez(save_filename, **np_dict)
   
def fsd_quantities(save_filename, device=torch.device('cuda'), cuda=True, NUM_EPOCH=500):
    data = np.load("/home/karenleung/Documents/simulator/lanechange/data/npz/lanechange_fancy.npz")
    data_np = data["data_np"]
    tl = data["tl"]
    key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
    N_data = len(tl)
    βs = []
    τs = []
    losses_fsd = []
    learning_rate = 0.01
    eps = 0.01
    features_stats = np.load("/home/karenleung/Documents/simulator/lanechange/scripts/stl_data/features_stats.npz")
    bt = np.load("/home/karenleung/Documents/simulator/lanechange/scripts/stl_data/τs.npz")
    best_τs = bt['τs']
    dy_μ, dy_σ = float(features_stats['dy_μ']), float(features_stats['dy_σ'])
    fsd_μ, fsd_σ = float(features_stats['fsd_μ']), float(features_stats['fsd_σ'])
    dy_thres = torch.tensor(3.5, dtype=torch.float64, device=device)
    A = LessThan("dy", ((dy_thres - dy_μ)/dy_σ))
    for idx in range(N_data):
        print("--------------------------- Index ", idx, "---------------------------")
        tl_ = int(tl[idx])
        car = data_np[idx,:tl_,:]
        key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
        # distance between front vehicles
        (dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right) = front_adjacent_car_gap(car, params, key_idx)
        # distance between rear vehicles
        (dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right) = rear_adjacent_car_gap(car, params, key_idx)
        # stopping distances between front vehiclesii
        (fsdf, fsdfl, fsdfr) = front_stopping_distance(car, params, key_idx)

        data_list = [dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right,
                     dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right,
                     fsdf, fsdfl, fsdfr,
                     car[:,key_idx['lane_index_ego']]]
        var_name = ['dxf', 'dyf', 'dxfl', 'dyfl', 'dxfr', 'dyfr',
                    'dxr', 'dyr', 'dxrl', 'dyrl', 'dxrr', 'dyrr',
                    'fsdf', 'fsdfl', 'fsdfr',
                    'lane_idx']

        (dxf, dyf, dxfl, dyfl, dxfr, dyfr, dxr, dyr, dxrl, dyrl, dxrr, dyrr, fsdf, fsdfl, fsdfr, li), episode = utils.convertArrayToEpisode(data_list, var_name)
        trace = stl.Trace(episode)
        lcd = lane_change_direction(car, key_idx)
        if lcd == "left":
            dy = dyfl
            fsd = fsdfl
        else:
            dy = dyfr
            fsd = fsdfr
        dy_trace_ = [trace.series[dy][-k-1]  for k in range(len(trace.series[dy]))]
        dy_trace = torch.tensor(dy_trace_, dtype=torch.float64, device=device).unsqueeze(0)
        fsd_trace_ = [trace.series[fsd][-k-1]  for k in range(len(trace.series[fsd]))]
        fsd_trace = torch.tensor(fsd_trace_, dtype=torch.float64, device=device).unsqueeze(0)
        print(len(fsd_trace[0]))
        dy_trace_std = (dy_trace - dy_μ)/dy_σ
        fsd_trace_std = (fsd_trace - fsd_μ)/fsd_σ
        best_τ = best_τs[idx]
        print("Best τ: ", best_τ)
        β = torch.randn(1,  dtype=torch.float64, requires_grad=True, device=device)
        B_fsd = GreaterThan("fsd", β)    
        inner_fsd = And(A, B_fsd)
        tin_fsd = Always(stl.Interval(0, best_τ))
        tout_fsd = Eventually()
        outer_fsd = TemporalComposition(tin_fsd, tout_fsd)
        fsd_formula = STLModel(inner_fsd, outer_fsd)

        for epoch in range(NUM_EPOCH):
            scale = scale_schedule(epoch, NUM_EPOCH)
            loss_fsd = torch.abs(fsd_formula(dy_trace_std, fsd_trace_std, scale=scale, cuda=cuda))
            loss_fsd.backward()
            with torch.no_grad():
                if epoch % 50 == 0:
                    print("Epoch number: ", epoch,
                          "\t β is : ", round(β.cpu().data.numpy()[0],3),
                          "\t Gradient is: ", round(β.grad.cpu().data.numpy()[0], 3),
                          "\t Loss is: ", round(loss_fsd.cpu().data.numpy()[0][0], 3),
                          "\t Scale is: ", scale)
                β -= learning_rate * (β.grad + 0.01*torch.randn(1,  dtype=torch.float64, device=device))
                β.grad.zero_()
        print("Epoch number: ", epoch,
          "\t β is : ", round(β.cpu().data.numpy()[0],3),
          "\t Gradient is: ", round(β.grad.cpu().data.numpy()[0], 3),
          "\t Loss is: ", round(loss_fsd.cpu().data.numpy()[0][0], 3),
          "\t Scale is: ", round(scale, 3))

        losses_fsd.append(loss_fsd.cpu().data.numpy())
        βs.append(β.cpu().data.numpy())
        τs.append(best_τ)

        del fsd_formula
        del B_fsd
        del β
        del inner_fsd
        del tin_fsd
        del tout_fsd
        del outer_fsd

    np_dict = {'τs': τs, 'βs': βs}
    np.savez(save_filename, **np_dict)


   



def cot_quantities(save_filename, device=torch.device('cuda'), cuda=True, rear=True, NUM_EPOCH=500):
    data = np.load("/home/karenleung/Documents/simulator/lanechange/data/npz/lanechange_fancy.npz")
    data_np = data["data_np"]
    tl = data["tl"]
    key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
    N_data = len(tl)
    γs = []
    τs = []
    losses_cot = []
    learning_rate = 0.01
    eps = 0.01
    features_stats = np.load("/home/karenleung/Documents/simulator/lanechange/scripts/stl_data/features_stats.npz")
    bt = np.load("/home/karenleung/Documents/simulator/lanechange/scripts/stl_data/τs.npz")
    best_τs = bt['τs']
    dy_μ, dy_σ = float(features_stats['dy_μ']), float(features_stats['dy_σ'])
    cotr_μ, cotr_σ = float(features_stats['cotr_μ']), float(features_stats['cotr_σ'])
    cotf_μ, cotf_σ = float(features_stats['cotf_μ']), float(features_stats['cotf_σ'])
    dy_thres = torch.tensor(3.5, dtype=torch.float64, device=device)
    A = LessThan("dy", ((dy_thres - dy_μ)/dy_σ))
    for idx in range(N_data):
        print("--------------------------- Index ", idx, "---------------------------")
        tl_ = int(tl[idx])
        car = data_np[idx,:tl_,:]
        key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
        # distance between front vehicles
        (dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right) = front_adjacent_car_gap(car, params, key_idx)
        # distance between rear vehicles
        (dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right) = rear_adjacent_car_gap(car, params, key_idx)
        # stopping distances between front vehiclesii
        (fsdf, fsdfl, fsdfr) = front_stopping_distance(car, params, key_idx)
        # crossover time between rear adjacent and fron vehicles
        (cotf, cotrl, cotrr) = crossover_time_rear(car, key_idx)
        (cotf, cotfl, cotfr) = crossover_time_rear(car, key_idx, rear=False)


        data_list = [dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right,
                     dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right,
                     fsdf, fsdfl, fsdfr,
                     cotf, cotrl, cotrr,
                     cotf, cotfl, cotfr,
                     car[:,key_idx['lane_index_ego']]]

        var_name = ['dxf', 'dyf', 'dxfl', 'dyfl', 'dxfr', 'dyfr',
                    'dxr', 'dyr', 'dxrl', 'dyrl', 'dxrr', 'dyrr',
                    'fsdf', 'fsdfl', 'fsdfr',
                    'cotf', 'cotrl', 'cotrr',
                    'cotf', 'cotfl', 'cotfr',
                    'lane_idx']


        (dxf, dyf, dxfl, dyfl, dxfr, dyfr, dxr, dyr, dxrl, dyrl, dxrr, dyrr, fsdf, fsdfl, fsdfr, cotf, cotrl, cotrr, cotf, cotfl, cotfr, li), episode = utils.convertArrayToEpisode(data_list, var_name)

        trace = stl.Trace(episode)
        lcd = lane_change_direction(car, key_idx)
        if lcd == "left":
            dy = dyfl
            if rear:
                cot = cotrl
            else:
                cot = cotfl
        else:
            dy = dyfr
            if rear:
                cot = cotrr
            else:
                cot = cotfr
        if rear:
            cot_μ = cotr_μ
            cot_σ = cotr_σ
        else:
            cot_μ = cotf_μ
            cot_σ = cotf_σ
            
        dy_trace_ = [trace.series[dy][-k-1]  for k in range(len(trace.series[dy]))]
        dy_trace = torch.tensor(dy_trace_, dtype=torch.float64, device=device).unsqueeze(0)
        cot_trace_ = [trace.series[cot][-k-1]  for k in range(len(trace.series[cot]))]
        cot_trace = torch.tensor(cot_trace_, dtype=torch.float64, device=device).unsqueeze(0)
        print(len(cot_trace[0]))
        dy_trace_std = (dy_trace - dy_μ)/dy_σ
        cot_trace_std = (cot_trace - cot_μ)/cot_σ
        best_τ = best_τs[idx]
        print("Best τ: ", best_τ)
        γ = torch.randn(1,  dtype=torch.float64, requires_grad=True, device=device)
        B_cot = GreaterThan("cot", γ)    
        inner_cot = And(A, B_cot)
        tin_cot = Always(stl.Interval(0, best_τ))
        tout_cot = Eventually()
        outer_cot = TemporalComposition(tin_cot, tout_cot)
        cot_formula = STLModel(inner_cot, outer_cot)

        for epoch in range(NUM_EPOCH):
            scale = scale_schedule(epoch, NUM_EPOCH)
            loss_cot = torch.abs(cot_formula(dy_trace_std, cot_trace_std, scale=scale, cuda=cuda))
            loss_cot.backward()
            with torch.no_grad():
                if epoch % 50 == 0:
                    print("Epoch number: ", epoch,
                          "\t γ is : ", round(γ.cpu().data.numpy()[0],3),
                          "\t Gradient is: ", round(γ.grad.cpu().data.numpy()[0], 3),
                          "\t Loss is: ", round(loss_cot.cpu().data.numpy()[0][0], 3),
                          "\t Scale is: ", scale)
                γ -= learning_rate * (γ.grad + 0.01*torch.randn(1,  dtype=torch.float64, device=device))

                γ.grad.zero_()
        print("Epoch number: ", epoch,
          "\t γ is : ", round(γ.cpu().data.numpy()[0],3),
          "\t Gradient is: ", round(γ.grad.cpu().data.numpy()[0], 3),
          "\t Loss is: ", round(loss_cot.cpu().data.numpy()[0][0], 3),
          "\t Scale is: ", round(scale, 3))

        losses_cot.append(loss_cot.cpu().data.numpy())
        γs.append(γ.cpu().data.numpy())
        τs.append(best_τ)

        del cot_formula
        del B_cot
        del γ
        del inner_cot
        del tin_cot
        del tout_cot
        del outer_cot


    np_dict = {'τs': τs, 'γs': γs}
    np.savez(save_filename, **np_dict)

def linear_quantities(save_filename, device=torch.device('cuda'), cuda=True, NUM_EPOCH=500):
    data = np.load("/home/karenleung/Documents/simulator/lanechange/data/npz/lanechange_fancy.npz")
    data_np = data["data_np"]
    tl = data["tl"]
    key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
    N_data = len(tl)
    δs = []
    ds = []
    losses = []
    learning_rate = 0.01
    eps = 0.01
    features_stats = np.load("/home/karenleung/Documents/simulator/lanechange/scripts/stl_data/features_stats.npz")
    bt = np.load("/home/karenleung/Documents/simulator/lanechange/scripts/stl_data/τs.npz")
    best_τs = bt['τs']


    dy_μ, dy_σ = float(features_stats['dy_μ']), float(features_stats['dy_σ'])  
    fsd_μ, fsd_σ = float(features_stats['fsd_μ']), float(features_stats['fsd_σ'])  
    cotr_μ, cotr_σ = float(features_stats['cotr_μ']), float(features_stats['cotr_σ'])  
    cotf_μ, cotf_σ = float(features_stats['cotf_μ']), float(features_stats['cotf_σ'])  
    dxf_μ, dxf_σ = float(features_stats['dxf_μ']), float(features_stats['dxf_σ'])  
    dxr_μ, dxr_σ = float(features_stats['dxr_μ']), float(features_stats['dxr_σ'])  
    fsdf_μ, fsdf_σ = float(features_stats['fsdf_μ']), float(features_stats['fsdf_σ'])  
    cotff_μ, cotff_σ = float(features_stats['cotff_μ']), float(features_stats['cotff_σ']) 

    dy_thres = torch.tensor(3.5, dtype=torch.float64, device=device)
    A = LessThan("dy", ((dy_thres - dy_μ)/dy_σ))
    for idx in range(N_data):
        print("--------------------------- Index ", idx, "---------------------------")
        tl_ = int(tl[idx])
        car = data_np[idx,:tl_,:]
        key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}
        # distance between front vehicles
        (dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right) = front_adjacent_car_gap(car, params, key_idx)
        # distance between rear vehicles
        (dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right) = rear_adjacent_car_gap(car, params, key_idx)
        # stopping distances between front vehiclesii
        (fsdf, fsdfl, fsdfr) = front_stopping_distance(car, params, key_idx)
        # crossover time between rear adjacent and fron vehicles
        (cotff, cotrl, cotrr) = crossover_time_rear(car, key_idx)
        (cotff, cotfl, cotfr) = crossover_time_rear(car, key_idx, rear=False)


        data_list = [dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right,
                     dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right,
                     fsdf, fsdfl, fsdfr,
                     cotff, cotrl, cotrr,
                     cotff, cotfl, cotfr,
                     car[:,key_idx['lane_index_ego']]]

        var_name = ['dxf', 'dyf', 'dxfl', 'dyfl', 'dxfr', 'dyfr',
                    'dxr', 'dyr', 'dxrl', 'dyrl', 'dxrr', 'dyrr',
                    'fsdf', 'fsdfl', 'fsdfr',
                    'cotff', 'cotrl', 'cotrr',
                    'cotff', 'cotfl', 'cotfr',
                    'lane_idx']


        (dxf, dyf, dxfl, dyfl, dxfr, dyfr, dxr, dyr, dxrl, dyrl, dxrr, dyrr, fsdf, fsdfl, fsdfr, cotf, cotrl, cotrr, cotff, cotfl, cotfr, li), episode = utils.convertArrayToEpisode(data_list, var_name)

        trace = stl.Trace(episode)
        lcd = lane_change_direction(car, key_idx)

        if lcd == "left":
            dy = dyfl
            fsd = fsdfl
            cotr = cotrl
            cotf = cotfl
        else:
            dy = dyfr
            fsd = fsdfr
            cotr = cotrr
            cotf = cotfr
                

        dy_trace_ = [trace.series[dy][-k-1]  for k in range(len(trace.series[dy]))]
        dy_trace = torch.tensor(dy_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        cotf_trace_ = [trace.series[cotf][-k-1]  for k in range(len(trace.series[cotf]))]
        cotf_trace = torch.tensor(cotf_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        cotr_trace_ = [trace.series[cotr][-k-1]  for k in range(len(trace.series[cotr]))]
        cotr_trace = torch.tensor(cotr_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        fsd_trace_ = [trace.series[fsd][-k-1]  for k in range(len(trace.series[fsd]))]
        fsd_trace = torch.tensor(fsd_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        dxf_trace_ = [trace.series[dxf][-k-1]  for k in range(len(trace.series[dxf]))]
        dxf_trace = torch.tensor(dxf_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        dxr_trace_ = [trace.series[dxr][-k-1]  for k in range(len(trace.series[dxr]))]
        dxr_trace = torch.tensor(dxr_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        fsdf_trace_ = [trace.series[fsdf][-k-1]  for k in range(len(trace.series[fsd]))]
        fsdf_trace = torch.tensor(fsdf_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        cotff_trace_ = [trace.series[cotff][-k-1]  for k in range(len(trace.series[fsd]))]
        cotff_trace = torch.tensor(cotff_trace_, dtype=torch.float64, device=device).unsqueeze(0)

        
        
        dy_trace_std = (dy_trace - dy_μ)/dy_σ
        cotf_trace_std = (cotf_trace - cotf_μ)/cotf_σ
        cotr_trace_std = (cotr_trace - cotr_μ)/cotr_σ
        fsd_trace_std = (fsd_trace - fsd_μ)/fsd_σ
        dxf_trace_std = (dxf_trace - dxf_μ)/dxf_σ
        dxr_trace_std = (dxr_trace - dxr_μ)/dxr_σ
        fsdf_trace_std = (fsdf_trace - fsdf_μ)/fsdf_σ
        cotff_trace_std = (cotff_trace - cotff_μ)/cotff_σ
        
        X = torch.stack([dxf_trace_std, dxr_trace_std, fsdf_trace_std, cotff_trace_std]).squeeze()
        V = torch.stack([cotf_trace_std, cotr_trace_std, fsd_trace_std]).squeeze()
        best_τ = best_τs[idx]
        print("Best τ: ", best_τ)
        # ALWAYS
        δ = torch.randn(1, 4,  dtype=torch.float64, requires_grad=True, device=device)
        d = torch.randn(1,  dtype=torch.float64, requires_grad=True, device=device)
        
        Cin = GreaterThan("ρ", d)
        Cout = Always()
        C = STLModel(Cin, Cout)
    #     break
    #     # EVENTUALLY ALWAYS
    #     B = GreaterThan("cot", γ)    
    #     inner_cot = And(A, B_cot)
    #     tin_cot = Always(stl.Interval(0, best_τ))
    #     tout_cot = Eventually()
    #     outer_cot = TemporalComposition(tin_cot, tout_cot)
    #     cot_formula = STLModel(inner_cot, outer_cot)

        for epoch in range(NUM_EPOCH):
            ρ = torch.matmul(δ, X)
            scale = scale_schedule(epoch, NUM_EPOCH)
            loss = torch.abs(C(ρ, scale=scale, cuda=cuda))
            loss.backward()
            with torch.no_grad():
                if epoch % 100 == 0:
                    print("Epoch number: ", epoch,
                          "\t δ is : ", δ.cpu().data.numpy()[0],
                          "\t d is : ", round(d.cpu().data.numpy()[0],3),
                          "\t Gradient δ is: ", δ.grad.cpu().data.numpy()[0],
                          "\t Gradient d is: ", round(d.grad.cpu().data.numpy()[0], 3),
                          "\t Loss is: ", round(loss.cpu().data.numpy()[0][0], 3),
                          "\t Scale is: ", scale)
                δ -= learning_rate * (δ.grad + 0.01*torch.randn(1,  dtype=torch.float64, device=device))
                δ.grad.zero_()
                d -= learning_rate * (d.grad + 0.01*torch.randn(1,  dtype=torch.float64, device=device))
                d.grad.zero_()
                
        print("Epoch number: ", epoch,
          "\t δ is : ", δ.cpu().data.numpy()[0],
          "\t d is : ", round(d.cpu().data.numpy()[0],3),
          "\t Gradient δ is: ", δ.grad.cpu().data.numpy()[0],
          "\t Gradient d is: ", round(d.grad.cpu().data.numpy()[0], 3),
          "\t Loss is: ", round(loss.cpu().data.numpy()[0][0], 3),
          "\t Scale is: ", round(scale, 3))

        losses.append(loss.cpu().data.numpy())
        δs.append(δ.cpu().data.numpy())
        ds.append(d.cpu().data.numpy())

        del δ
        del d
        del Cin
        del Cout
        del C


        np_dict = {'δs': δs, 'ds': ds}
        np.savez(save_filename, **np_dict)

def compute_mean_var_features(data_np, tl):
    dys = []
    dxfs = []
    dxrs = []
    fsds = []
    fsdfs = []
    cotrs = []
    cotfs = []
    cotffs = []
    N_data = data_np.shape[0]
    for idx in range(N_data):
        if idx % 200 == 0:
            print("Index: ", idx)
        car = data_np[idx,:int(tl[idx]),:]
        key_idx = {k: i for (i,k) in enumerate(default_dict.keys())}

        # distance between front vehicles
        (dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right) = front_adjacent_car_gap(car, params, key_idx)
        # distance between rear vehicles
        (dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right) = rear_adjacent_car_gap(car, params, key_idx)
        # stopping distances between front vehiclesii
        (fsdf, fsdfl, fsdfr) = front_stopping_distance(car, params, key_idx)
        # crossover time between rear adjacent and fron vehicles
        (cotf, cotrl, cotrr) = crossover_time_rear(car, key_idx)
        (cotf, cotfl, cotfr) = crossover_time_rear(car, key_idx, rear=False)


        data_list = [dx_front, dy_front, dx_front_left, dy_front_left, dx_front_right, dy_front_right,
                     dx_rear, dy_rear, dx_rear_left, dy_rear_left, dx_rear_right, dy_rear_right,
                     fsdf, fsdfl, fsdfr,
                     cotf, cotrl, cotrr,
                     cotf, cotfl, cotfr,
                     car[:,key_idx['lane_index_ego']]]
        var_name = ['dxf', 'dyf', 'dxfl', 'dyfl', 'dxfr', 'dyfr',
                    'dxr', 'dyr', 'dxrl', 'dyrl', 'dxrr', 'dyrr',
                    'fsdf', 'fsdfl', 'fsdfr',
                    'cotf', 'cotrl', 'cotrr',
                    'cotf', 'cotfl', 'cotfr',
                    'lane_idx']

        (dxf, dyf, dxfl, dyfl, dxfr, dyfr, dxr, dyr, dxrl, dyrl, dxrr, dyrr, fsdf, fsdfl, fsdfr, cotf, cotrl, cotrr, cotf, cotfl, cotfr, li), episode = utils.convertArrayToEpisode(data_list, var_name)
        trace = stl.Trace(episode)
        lcd = lane_change_direction(car, key_idx)
        # x_np = [trace.series[dyfl][-k-1]  for k in range(len( trace.series[dyfl]))]
        if lcd == "left":
            dy = dyfl
            fsd = fsdfl
            cotr = cotrl
            cotf = cotfl
        else:
            dy = dyfr
            fsd = fsdfr
            cotr = cotrr
            cotf = cotfr


        xx = [trace.series[dy][-k-1]  for k in range(len(trace.series[dy]))]
        yy = [trace.series[fsd][-k-1]  for k in range(len(trace.series[fsd]))]
        zz = [trace.series[cotr][-k-1]  for k in range(len(trace.series[cotr]))]
        ww = [trace.series[cotf][-k-1]  for k in range(len(trace.series[cotf]))]
        
        aa = [trace.series[dxf][-k-1]  for k in range(len(trace.series[dxf]))]
        bb = [trace.series[dxr][-k-1]  for k in range(len(trace.series[dxr]))]
        cc = [trace.series[fsdf][-k-1]  for k in range(len(trace.series[fsdf]))]
        dd = [trace.series[cotf][-k-1]  for k in range(len(trace.series[cotf]))]
        
        
        
        for (x,y,z,w) in zip(xx,yy,zz,ww):
            dys.append(x)
            fsds.append(y)
            cotrs.append(z)
            cotfs.append(w)
        for (a,b,c,d) in zip(aa,bb,cc,dd):
            dxfs.append(a)
            dxrs.append(b)
            fsdfs.append(c)
            cotffs.append(d)
            
    dy_μ, dy_σ = np.mean(dys), np.std(dys)
    fsd_μ, fsd_σ = np.mean(fsds), np.std(fsds)
    cotr_μ, cotr_σ = np.mean(cotrs), np.std(cotrs)
    cotf_μ, cotf_σ = np.mean(cotfs), np.std(cotfs)
    
    dxf_μ, dxf_σ = np.mean(dxfs), np.std(dxfs)
    dxr_μ, dxr_σ = np.mean(dxrs), np.std(dxrs)
    fsdf_μ, fsdf_σ = np.mean(fsdfs), np.std(fsdfs)
    cotff_μ, cotff_σ = np.mean(cotffs), np.std(cotffs)
    
    
    return (dy_μ, dy_σ), (fsd_μ, fsd_σ), (cotr_μ, cotr_σ), (cotf_μ, cotf_σ), (dxf_μ, dxf_σ), (dxr_μ, dxr_σ), (fsdf_μ, fsdf_σ), (cotff_μ, cotff_σ)
# if __name__=="__main__":



