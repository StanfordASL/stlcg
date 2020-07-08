# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import sys
import IPython
# from utils import tensor_to_str
'''
Important information:
- This has the option to use an arithmetic-geometric mean robustness metric: https://arxiv.org/pdf/1903.05186.pdf. The default is not to use it. But this is still being tested.
- Assume inputs are already reversed, but user does not need to worry about the indexing.
- "pscale" stands for "predicate scale" (not the scale used in maxish and minish)
- "scale" is the scale used in maxish and minish which Always, Eventually, Until, and Then uses.
- "time" variable when computing robustness: time=0 means the current time, t=1 means next time step. The reversal of the trace is accounted for inside the function, the user does not need to worry about this
- must specify subformula (no default string value)
'''


# TODO:
# - Run tests to ensure that "Expression" correctly overrides operators
# - Make a test for each temporal operator, and make sure that they all produce the expected output for at least one example trace
# - Implement log-barrier
# - option to choose type of padding

LARGE_NUMBER = 1E4

def tensor_to_str(tensor):
    '''
    turn tensor into a string for printing
    '''
    device = tensor.device.type
    req_grad = tensor.requires_grad
    if req_grad == False:
        return "input"
    tensor = tensor.detach()
    if device == "cuda":
        tensor = tensor.cpu()
    return str(tensor.numpy())

def convert_to_input_values(inputs):
    x_, y_ = inputs
    if isinstance(x_, Expression):
        assert x_.value is not None, "Input Expression does not have numerical values"
        x_ret = x_.value
    elif isinstance(x_, torch.Tensor):
        x_ret = x_
    elif isinstance(x_, tuple):
        x_ret = convert_to_input_values(x_)
    else:
        raise ValueError("First argument is an invalid input trace")

    if isinstance(y_, Expression):
        assert y_.value is not None, "Input Expression does not have numerical values"
        y_ret = y_.value
    elif isinstance(y_, torch.Tensor):
        y_ret = y_
    elif isinstance(y_, tuple):
        y_ret = convert_to_input_values(y_)
    else:
        raise ValueError("Second argument is an invalid input trace")

    return (x_ret, y_ret)
    

class Maxish(torch.nn.Module):
    '''
    Function to compute the max, or softmax, or other variants of the max function.
    '''
    def __init__(self, name="Maxish input"):
        super(Maxish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1, keepdim=True, agm=False, distributed=False):
        '''
        x is of size [batch_size, N, ...] where N is typically the trace length.

        if scale <= 0, then the true max is used, otherwise, the softmax is used.

        dim is the dimension which the max is applied over. Default: 1

        keepdim keeps the dimension of the input tensor. Default: True

        agm is the arithmetic-geometric mean. Currently in progress. If some elements are >0, output is the average of those elements. If all the elements <= 0, output is -ᵐ√(Πᵢ (1 - ηᵢ)) + 1. scale doesn't play a role here except to switch between the using the AGM or true robustness value (scale <=0). Default: False
        
        distributed addresses the case when there are multiple max values. As max is poorly defined in these cases, PyTorch (randomly?) selects one of the max values only. If distributed=True and scale <=0 then it will average over the max values and split the gradients equally. Default: False
        '''

        if isinstance(x, Expression):
            assert x.value is not None, "Input Expression does not have numerical values"
            x = x.value
        if scale > 0:
            if agm == True:
                if torch.gt(x, 0).any():
                    return x[torch.gt(x, 0)].reshape(*x.shape[:-1], -1).mean(dim=dim, keepdim=keepdim)
                else:
                    return -torch.log(1-x).mean(dim=dim, keepdim=keepdim).exp() + 1
            else:
                # return (torch.softmax(x*scale, dim=dim)*x).sum(dim, keepdim=keepdim)
                return torch.log(torch.exp(x*scale).sum(dim=dim, keepdim=keepdim))/scale
        else:
            if distributed:
                return self.distributed_true_max(x, dim=dim, keepdim=keepdim)
            else:
                return x.max(dim, keepdim=keepdim)[0]


    @staticmethod
    def distributed_true_max(xx, dim=1, keepdim=True):
        m, mi = torch.max(xx, dim, keepdim=True)
        inds = xx == m
        return torch.where(inds, xx, xx*0).sum(dim, keepdim=keepdim) / inds.sum(dim, keepdim=keepdim)


    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [str(self.input_name)]

class Minish(torch.nn.Module):
    '''
    Function to compute the min, or softmin, or other variants of the min function.
    '''
    def __init__(self, name="Minish input"):
        super(Minish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1, keepdim=True, agm=False, distributed=False):
        '''
        x is of size [batch_size, N, ...] where N is typically the trace length.

        if scale <= 0, then the true max is used, otherwise, the softmax is used.

        dim is the dimension which the max is applied over. Default: 1

        keepdim keeps the dimension of the input tensor. Default: True

        agm is the arithmetic-geometric mean. Currently in progress. If all elements are >0, output is ᵐ√(Πᵢ (1 + ηᵢ)) - 1.If some the elements <= 0, output is the average of those negative values. scale doesn't play a role here except to switch between the using the AGM or true robustness value (scale <=0).

        distributed addresses the case when there are multiple max values. As max is poorly defined in these cases, PyTorch (randomly?) selects one of the max values only. If distributed=True and scale <=0 then it will average over the max values and split the gradients equally. Default: False
        '''

        if isinstance(x, Expression):
            assert x.value is not None, "Input Expression does not have numerical values"
            x = x.value

        if scale > 0:
            if agm == True:
                if torch.gt(x, 0).all():
                    return torch.log(1+x).mean(dim=dim, keepdim=keepdim).exp() - 1
                else:
                    # return x[torch.lt(x, 0)].reshape(*x.shape[:-1], -1).mean(dim=dim, keepdim=keepdim)
                    return  (torch.lt(x,0) * x).sum(dim, keepdim=keepdim) / torch.lt(x, 0).sum(dim, keepdim=keepdim)
            else:
                # return (torch.softmax(-x*scale, dim=dim)*x).sum(dim, keepdim=keepdim)
                return -torch.log(torch.exp(-x*scale).sum(dim=dim, keepdim=keepdim))/scale

        else:
            if distributed:
                return self.distributed_true_min(x, dim=dim, keepdim=keepdim)
            else:
                return x.min(dim, keepdim=keepdim)[0]

    @staticmethod
    def distributed_true_min(xx, dim=1, keepdim=True):
        m, mi = torch.min(xx, dim, keepdim=True)
        inds = xx == m
        return torch.where(inds, xx, xx*0).sum(dim, keepdim=keepdim) / inds.sum(dim, keepdim=keepdim)


    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [str(self.input_name)]



class STL_Formula(torch.nn.Module):
    '''
    All STL formulas have the following functions:
    robustness_trace: Computes the robustness trace using the method outlined in WAFR20 paper
    robustness: Gets the last (since signal is reversed) entry of the robustness trace
    eval_trace: Computes the robustness trace and returns True is the robustness value is > 0
    eval: Gets the last (since signal is reversed) entry of the eval trace
    forward: The forward function of this STL_formula PyTorhc module (default to the robustness_trace function)

    inputs to these functions:
    trace: the input signal. If the formula has two subformulas (e.g., And), then it is a tuple of the two inputs. An input can be a tensor of size [batch_size, time_dim,...], of an Expression with a ._val (Tensor) associated with the expression.
    pscale: predicate scale. Default: 1
    scale: scale for the max/min function.  Default: -1
    keepdim: Output shape is the same as the input tensor shapes. Default: True
    agm: Use arithmetic-geometric mean. (In progress.) Default: False
    distributed: Use the distributed mean. Default: False
    '''

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, trace, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, inputs, time=0, pscale=1, scale=-1, keepdim=True, agm=False,distributed=False, **kwargs):
        '''
        Extracts the robustness_trace value at the given time. (Default: time=0 from the user's perspective. But will be the taken from the end since the trace is reversed.)
        '''
        return self.robustness_trace(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)[:,-(time+1),:].unsqueeze(1)

    def eval_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        '''
        The values in eval_trace are 0 or 1 (False or True)
        '''
        return self.robustness_trace(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs) > 0

    def eval(self, inputs, time=0, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        '''
        Extracts the eval_trace value at the given time. (Default: time=0 from the user's perspective. But will be the taken from the end since the trace is reversed.)
        '''
        return self.eval_trace(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)[:,-(time+1),:].unsqueeze(1)                 # [batch_size, time_dim, x_dim]

    def forward(formula, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        '''
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.
        '''
        if isinstance(inputs, Expression):
            assert inputs.value is not None, "Input Expression does not have numerical values"
            return formula.robustness_trace(inputs.value, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        elif isinstance(inputs, torch.Tensor):
            return formula.robustness_trace(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        elif isinstance(inputs, tuple):
            return formula.robustness_trace(convert_to_input_values(inputs), pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        else:
            raise ValueError("Not a invalid input trace")

    def __str__(self):
        raise NotImplementedError("__str__ not yet implemented")

    def __and__(phi, psi):
        return And(phi, psi)

    def __or__(phi, psi):
        return Or(phi, psi)

    def __invert__(phi):
        return Negation(phi)


class Identity(STL_Formula):

    def __init__(self, name='x'):
        super(Identity, self).__init__()
        self.name = name

    def robustness_trace(self, trace, pscale=1, **kwargs):
        return trace * pscale 

    def _next_function(self):
        return []

    def __str__(self):
        return "%s" %self.name




class Temporal_Operator(STL_Formula):
    '''
    Class to compute Eventuall and Always. This builds an RNN cell (see WAFR20 paper)
    subformula: The formula that the temporal operator is applied to.
    interval: either None (defaults to [0, np.inf]), [a, b] ( b < np.inf), [a, np.inf] (a > 0)
    '''
    def __init__(self, subformula, interval=None):
        super(Temporal_Operator, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if self.interval is None else self.interval
        self.rnn_dim = 1 if not self.interval else self.interval[-1]    # rnn_dim=1 if interval is [0, ∞) otherwise rnn_dim=end of interval
        if self.rnn_dim == np.inf:
            self.rnn_dim = self.interval[0]
        self.steps = 1 if not self.interval else self.interval[-1] - self.interval[0] + 1   # steps=1 if interval is [0, ∞) otherwise steps=length of interval
        self.operation = None
        # Matrices that shift a vector and add a new entry at the end.
        self.M = torch.tensor(np.diag(np.ones(self.rnn_dim-1), k=1)).requires_grad_(False).float()
        self.b = torch.zeros(self.rnn_dim).unsqueeze(-1).requires_grad_(False).float()
        self.b[-1] = 1.0

    def _initialize_rnn_cell(self, x):

        '''
        x is [batch_size, time_dim, x_dim]
        initial rnn state is [batch_size, rnn_dim, x_dim]
        This requires padding on the signal. Currently, the default is to extend the last value. 
        TODO: have option on this padding

        The initial hidden state is of the form (hidden_state, count). count is needed just for the case with self.interval=[0, np.inf) and distributed=True. Since we are scanning through the sigal and outputing the min values incrementally, the distributed min function doesn't apply. If there are multiple min values along the signal, the gradient will be distributed equally across them. Otherwise it will only apply to the value that occurs earliest in the signal (i.e., last as we process the signal backwards).
        '''
        raise NotImplementedError("_initialize_rnn_cell is not implemented")

    def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False,**kwargs):
        '''
        x: rnn input [batch_size, 1, ...]
        h0: input rnn hidden state  (see WAFR20 paper) It depends on the interval chosen. Roughly in the form of [batch_size, rnn_dim,...]
        '''
        raise NotImplementedError("_initialize_rnn_cell is not implemented")


    def _run_rnn_cell(self, x, scale, agm=False, distributed=False):

        outputs = []
        states = []
        hc = self._initialize_rnn_cell(x)                                # [batch_size, rnn_dim, x_dim]
        xs = torch.split(x, 1, dim=1)                                    # time_dim tuple
        time_dim = len(xs)
        for i in range(time_dim):
            o, hc = self._rnn_cell(xs[i], hc, scale, agm=agm, distributed=distributed)
            outputs.append(o)
            states.append(hc)
            # states.append(hc[0])
        return outputs, states


    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        # Compute the robustness trace of the subformula and that is the input to the temporal operator graph.
        trace = self.subformula(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        outputs, states = self._run_rnn_cell(trace, scale=scale, agm=agm, distributed=distributed)
        return torch.cat(outputs, dim=1)                              # [batch_size, time_dim, ...]

    def _next_function(self):
        # next function is the input subformula
        return [self.subformula]



class Always(Temporal_Operator):
    def __init__(self, subformula, interval=None):
        super(Always, self).__init__(subformula=subformula, interval=interval)
        self.operation = Minish()
        self.oper = "min"

    def _initialize_rnn_cell(self, x):
        # init_val = LARGE_NUMBER
        if x.is_cuda:
            self.M = self.M.cuda()
            self.b = self.b.cuda()
        h0 = torch.ones([x.shape[0], self.rnn_dim, x.shape[2]], device=x.device)*x[:,:1,:]
        count = 0.0
        # if self.interval is [a, np.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == np.inf) & (self._interval[0] > 0):
            d0 = x[:,:1,:]
            return ((d0, h0.to(x.device)), count)

        return (h0.to(x.device), count)

    def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False, **kwargs):
        '''
        x: rnn input [batch_size, 1, ...]
        hc=(h0, c) h0 is the input rnn hidden state  [batch_size, rnn_dim, ...]. c is the count. Initialized by self._initialize_rnn_cell
        '''
        h0, c = hc
        if self.operation is None:
            raise Exception()
        # keeping track of all values that share the min value so the gradients can be distributed equally.
        if self.interval is None:
            if distributed:
                if x == h0:
                    new_h =  (h0 * c + x)/ (c + 1)
                    new_c = c + 1.0
                elif x < h0:
                    new_h = x
                    new_c = 1.0
                else:
                    new_h = h0
                    new_c = c
                state = (new_h, new_c)
                output = new_h
            else:
                input_ = torch.cat([h0, x], dim=1)                          # [batch_size, rnn_dim+1, x_dim]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm)       # [batch_size, 1, x_dim]
                state = (output, None)
        else: # is self.interval is [a, np.inf)
            if (self._interval[1] == np.inf) & (self._interval[0] > 0):
                d0, h0 = h0
                dh = torch.cat([d0, h0[:,:1,:]], dim=1)                             # [batch_size, 2, x_dim]
                output = self.operation(dh, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1, x_dim]
                state = ((output, torch.matmul(self.M, h0) + self.b * x), None)    
            else: # self.interval is [a, b]
                state = (torch.matmul(self.M, h0) + self.b * x, None)
                h0x = torch.cat([h0, x], dim=1)                             # [batch_size, rnn_dim+1, x_dim]
                input_ = h0x[:,:self.steps,:]                               # [batch_size, self.steps, x_dim]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1, x_dim]
        return output, state

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"

class Eventually(Temporal_Operator):
    def __init__(self, subformula='Eventually input', interval=None):
        super(Eventually, self).__init__(subformula=subformula, interval=interval)
        self.operation = Maxish()
        self.oper = "max"

    def _initialize_rnn_cell(self, x):

        # init_val = -LARGE_NUMBER
        if x.is_cuda:
            self.M = self.M.cuda()
            self.b = self.b.cuda()
        h0 = torch.ones([x.shape[0], self.rnn_dim, x.shape[2]], device=x.device)*x[:,:1,:]
        count = 0.0
        if (self._interval[1] == np.inf) & (self._interval[0] > 0):
            d0 = x[:,:1,:]
            return ((d0, h0.to(x.device)), count)
        return (h0.to(x.device), count)

    def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False, **kwargs):
        '''
        x: rnn input [batch_size, 1, ...]
        h0: input rnn hidden state  [batch_size, rnn_dim, ...]
        '''
        h0, c = hc
        if self.operation is None:
            raise Exception()

        if self.interval is None:
            if distributed:
                if x == h0:
                    new_h =  (h0 * c + x)/ (c + 1)
                    new_c = c + 1.0
                elif x > h0:
                    new_h = x
                    new_c = 1.0
                else:
                    new_h = h0
                    new_c = c
                state = (new_h, new_c)
                output = new_h
            else:
                input_ = torch.cat([h0, x], dim=1)                          # [batch_size, rnn_dim+1, x_dim]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm)       # [batch_size, 1, x_dim]
                state = (output, None)
        else:
            if (self._interval[1] == np.inf) & (self._interval[0] > 0):
                d0, h0 = h0
                dh = torch.cat([d0, h0[:,:1,:]], dim=1)                             # [batch_size, 2, x_dim]
                output = self.operation(dh, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1, x_dim]
                state = ((output, torch.matmul(self.M, h0) + self.b * x), None)    
            else:
                state = (torch.matmul(self.M, h0) + self.b * x, None)
                h0x = torch.cat([h0, x], dim=1)                             # [batch_size, rnn_dim+1, x_dim]
                input_ = h0x[:,:self.steps,:]                               # [batch_size, self.steps, x_dim]
                output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm, distributed=distributed)               # [batch_size, 1, x_dim]
        return output, state

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"

class LessThan(STL_Formula):
    '''
    lhs <= val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression 
    val can be a float, int, Expression, or tensor. It cannot be a string.
    '''
    def __init__(self, lhs='x', val='c'):
        super(LessThan, self).__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "value on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, pscale=1.0, **kwargs):
        if isinstance(self.val, Expression):
            return (self.val.value - trace)*pscale
        else:
            return (self.val - trace)*pscale

    def _next_function(self):
        # expects self.lhs to be a string (used for visualizing the graph)
        # if isinstance(self.lhs, Expression):
        #     return [self.lhs.name, self.val]
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name

        if isinstance(self.val, str): # could be a string if robustness_trace is never called
            return lhs_str + " <= " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " <= " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " <= " + tensor_to_str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " <= " + str(self.val)


class GreaterThan(STL_Formula):
    '''
    lhs >= val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression 
    val can be a float, int, Expression, or tensor. It cannot be a string.
    '''
    def __init__(self, lhs='x', val='c'):
        super(GreaterThan, self).__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "value on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, pscale=1.0, **kwargs):
        if isinstance(self.val, Expression):
            return (trace - self.val.value)*pscale
        else:
            return (trace - self.val)*pscale

    def _next_function(self):
        # expects self.lhs to be a string (used for visualizing the graph)
        # if isinstance(self.lhs, Expression):
        #     return [self.lhs.name, self.val]
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name

        if isinstance(self.val, str): # could be a string if robustness_trace is never called
            return lhs_str + " >= " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " >= " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " >= " + tensor_to_str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " >= " + str(self.val)

class Equal(STL_Formula):
    '''
    lhs == val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression 
    val can be a float, int, Expression, or tensor. It cannot be a string.
    '''
    def __init__(self, lhs='x', val='c'):
        super(Equal, self).__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "value on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, pscale=1.0, **kwargs):
        if isinstance(self.val, Expression):
            return -torch.abs(trace - self.val.value)*pscale

        return -torch.abs(trace - self.val)*pscale

    def _next_function(self):
        # if isinstance(self.lhs, Expression):
        #     return [self.lhs.name, self.val]
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name

        if isinstance(self.val, str): # could be a string if robustness_trace is never called
            return lhs_str + " == " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " == " + self.val.name
        if isinstance(self.val, torch.Tensor):
            return lhs_str + " == " + tensor_to_str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " == " + str(self.val)

class Negation(STL_Formula):
    '''
    not Subformula
    '''
    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, distributed=False, **kwargs):
        return -self.subformula(inputs, pscale=pscale, scale=scale, keepdim=keepdim, distributed=distributed, **kwargs)

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula]

    def __str__(self):
        return "¬(" + str(self.subformula) + ")"

class Implies(STL_Formula):
    '''
    Implies
    '''
    def __init__(self, subformula1, subformula2):
        super(Implies, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Maxish()


    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        xx = torch.stack([-trace1, trace2], dim=-1)      # [batch_size, time_dim, ..., 2]
        return self.operation(xx, scale, dim=-1, keepdim=keepdim, agm=agm, distributed=distributed)   # [batch_size, time_dim, ...]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") => (" + str(self.subformula2) + ")"

class And(STL_Formula):
    '''
    inputs: tuple (x,y) where x and y are the inputs to each subformula respectively. x or y can also be a tuple if the subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z) would have inputs=(x, (y,z)))    )
    trace1 and trace2 are size [batch_size, time_dim, x_dim]
    '''
    def __init__(self, subformula1, subformula2):
        super(And, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Minish()

    @staticmethod
    def separate_and(formula, input_, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        if formula.__class__.__name__ != "And":
            return formula(input_, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        else:
            return torch.cat([And.separate_and(formula.subformula1, input_[0], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs), And.separate_and(formula.subformula2, input_[1], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)], axis=-1)

    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        xx = torch.cat([And.separate_and(self.subformula1, inputs[0], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs), And.separate_and(self.subformula2, inputs[1], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)], axis=-1)
        return self.operation(xx, scale, dim=-1, keepdim=keepdim, agm=agm, distributed=distributed)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"

class Or(STL_Formula):
    '''
    inputs: tuple (x,y) where x and y are the inputs to each subformula respectively. x or y can also be a tuple if the subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z) would have inputs=(x, (y,z)))    )
    trace1 and trace2 are size [batch_size, time_dim, x_dim]
    '''
    def __init__(self, subformula1, subformula2):
        super(Or, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Maxish()

    @staticmethod
    def separate_or(formula, input_, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        if formula.__class__.__name__ != "Or":
            return formula(input_, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        else:
            return torch.cat([Or.separate_or(formula.subformula1, input_[0], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs), Or.separate_or(formula.subformula2, input_[1], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)], axis=-1)

    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        xx = torch.cat([Or.separate_or(self.subformula1, inputs[0], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs), Or.separate_or(self.subformula2, inputs[1], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)], axis=-1)
        return self.operation(xx, scale, dim=-1, keepdim=keepdim, agm=agm, distributed=distributed)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"

class Until(STL_Formula):
    def __init__(self, subformula1="Until subformula1", subformula2="Until subformula2", overlap=True):
        '''
        subformula1 U subformula2 (ϕ U ψ)
        This assumes that ϕ is always true before ψ becomes true.
        If overlap=True, then the last time step that ϕ is true, ψ starts being true. That is, sₜ ⊧ ϕ and sₜ ⊧ ψ.
        If overlap=False, when ϕ stops being true, ψ starts being true. That is sₜ ⊧ ϕ and sₜ+₁ ⊧ ψ, but sₜ ¬⊧ ψ
        '''
        super(Until, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        if overlap == False:
            self.subformula2 = Eventually(subformula=subformula2, interval=[0,1])


    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        '''
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        assert isinstance(self.subformula1, STL_Formula), "Subformula1 needs to be an stl formula"
        assert isinstance(self.subformula2, STL_Formula), "Subformula2 needs to be an stl formula"
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        Alw = Always(subformula=Identity(name=str(self.subformula1)))
        minish = Minish()
        maxish = Maxish()
        LHS = trace2.unsqueeze(-1).repeat([1, 1, 1,trace2.shape[1]]).permute(0, 3, 2, 1)                                  # [batch_size, time_dim, x_dim, time_dim]
        # TODO: make same as first/last value in the input.
        RHS = torch.ones_like(LHS)*-LARGE_NUMBER                                                    # [batch_size, time_dim, x_dim, time_dim]
        for i in range(trace2.shape[1]):
            RHS[:,i:,:,i] = Alw(trace1[:,i:,:], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        # first min over the (ρ(ψ), ◻ρ(ϕ))
        # then max over the t′ dimension (the second time_dim dimension)
        return maxish(minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1, keepdim=keepdim, agm=agm, distributed=distributed).squeeze(-1), scale=scale, dim=-1, keepdim=keepdim, agm=agm, distributed=distributed).squeeze(-1)                                                              # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + "(" + str(self.subformula2) + ")"

class Then(STL_Formula):
    '''
    subformula1 T subformula2
    '''
    def __init__(self, subformula1, subformula2, overlap=True):
        super(Then, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        if overlap == False:
            self.subformula2 = Eventually(subformula=subformula2, interval=[0,1])

    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        '''
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        assert not isinstance(self.subformula1, str), "Subformula1 needs to be an stl formula"
        assert not isinstance(self.subformula2, str), "Subformula2 needs to be an stl formula"
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        Ev = Eventually(subformula=Identity(name=str(self.subformula1)))
        minish = Minish()
        maxish = Maxish()
        LHS = trace2.unsqueeze(-1).repeat([1, 1, 1,trace2.shape[1]]).permute(0, 3, 2, 1)                                  # [batch_size, time_dim, x_dim, time_dim]
        # TODO: make same as first/last value in the input.
        RHS = torch.ones_like(LHS)*-LARGE_NUMBER                                                 # [batch_size, time_dim, x_dim, time_dim]
        for i in range(trace2.shape[1]):
            RHS[:,i:,:,i] = Ev(trace1[:,i:,:], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
        # first min over the (ρ(ψ), ◻ρ(ϕ))
        # then max over the t′ dimension (the second time_dim dimension)
        return maxish(minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1, keepdim=keepdim, agm=agm, distributed=distributed).squeeze(-1), scale=scale, dim=-1, keepdim=keepdim, agm=agm, distributed=distributed).squeeze(-1)                                                              # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " T " + "(" + str(self.subformula2) + ")"

class Integral1d(STL_Formula):
    def __init__(self, subformula, interval=None):
        super(Integral1d, self).__init__()
        self.subformula = subformula
        self.padding_size = None if interval is None else interval[1]
        self.interval = interval
        if interval is not None:
            kernel = interval[1] - interval[0] + 1
            self.conv = torch.nn.Conv1d(1, 1, kernel, padding=0, bias=False)
            for param in self.conv.parameters():
                param.requires_grad = False
            self.conv.weight /= self.conv.weight
            


    def construct_padding(self, padding_type, custom_number=100):
        if self.padding_size is not None:
            if padding_type == "zero":
                return torch.zeros([1, self.padding_size, 1])
            elif padding_type == "custom":
                return torch.ones([1, self.padding_size, 1])*custom_number
            elif padding_type == "same":
                return torch.ones([1, self.padding_size, 1])
        else:
            return None

    def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=False, use_relu=False, padding_type="same", custom_number=100, integration_scheme="riemann", **kwargs):
        
        subformula_trace = self.subformula(inputs, pscale=pscale, scale=scale, keepdim=keepdim, **kwargs)


        if self.interval is not None:
            if integration_scheme == "trapz":
                self.conv.weight[:,:,0] /= 2
                self.conv.weight[:,:,-1] /= 2
            padding = self.construct_padding(padding_type, custom_number)
            if subformula_trace.is_cuda:
                padding = padding.cuda()
            if padding_type == "same":
                padding = padding * subformula_trace[:,0,:]
            signal = torch.cat([padding, subformula_trace], dim=1).transpose(1,2)
            if use_relu:
                return self.conv(torch.relu(signal)).transpose(1,2)[:,:subformula_trace.shape[1],:]
            else:
                return self.conv(signal).transpose(1,2)[:,:subformula_trace.shape[1],:]
        else:
            if integration_scheme == "trapz":
                pad = torch.zeros_like(subformula_trace[:,:1,:])
                if subformula_trace.is_cuda:
                    pad = padding.cuda()
                signal = torch.cat([pad, (subformula_trace[:,:-1,:] + subformula_trace[:,1:,:])/2], dim=1)
            else:
                signal = subformula_trace
            if use_relu == True:
                return torch.cumsum(torch.relu(signal), dim=1)
            else:
                return torch.cumsum(signal, dim=1)

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula]

    def __str__(self):
        return "I" + str(self.interval) + "(" + str(self.subformula) + ")"



class Expression(torch.nn.Module):
    '''
    Wraps a pytorch arithmetic operation, so that we can intercept and overload comparison operators.
    Expression allows us to express tensors using their names to make it easier to code up and read, 
    but also keep track of their numeric values.
    '''
    def __init__(self, name, value):
        super(Expression, self).__init__()
        self.name = name
        self.value = value
        
    def set_name(self, new_name):
        self.name = new_name
        
    def set_value(self, new_value):
        self.value = new_value

    def __neg__(self):
        return Expression(-self.value)
    
    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + '+' + other.name, self.value + other.value)
        else:
            return Expression(self.name + "+other", self.value + other)

    def __radd__(self, other):
        return self.__add__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular add 

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + '-' + other.name, self.value - other.value)
        else:
            return Expression(self.name + "-other", self.value - other)


    def __rsub__(self, other):
        return Expression(other - self.value)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + '*' + other.name, self.value * other.value)
        else:
            return Expression(self.name + "*other", self.value * other)


    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        # This is the new form required by Python 3
        numerator = a
        denominator = b
        num_name = 'num'
        denom_name = 'denom'
        if isinstance(numerator, Expression):
            num_name = numerator.name
            numerator = numerator.value
        if isinstance(denominator, Expression):
            denom_name = denominator.name
            denominator = denominator.value
        return Expression(num_name + '/' + denom_name, numerator/denominator)
        
    # Comparators
    def __lt__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs) 
        # if isinstance(lhs, Expression) and isinstance(rhs, Expression):
        #     return LessThan(lhs, rhs) 
        # elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
        #     # This case cannot occur. If neither is an Expression, why are you calling this method?
        #     raise Exception('What are you doing?')
        # elif not isinstance(rhs, Expression):
        #     return LessThan(lhs, rhs)
        # elif not isinstance(lhs, Expression):
        #     assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of LessThan needs to be a string or Expression"
        #     return LessThan(lhs, rhs.value)

    def __le__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)
        # if isinstance(lhs, Expression) and isinstance(rhs, Expression):
        #     return LessThan(lhs.name, rhs) 
        # elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
        #     # This case cannot occur. If neither is an Expression, why are you calling this method?
        #     raise Exception('What are you doing?')
        # elif not isinstance(rhs, Expression):
        #     return LessThan(lhs.name, rhs)
        # elif not isinstance(lhs, Expression):
        #     assert type(lhs)==str, "LHS of LessThan needs to be a string"
        #     return LessThan(lhs, rhs)

    def __gt__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)
        # if isinstance(lhs, Expression) and isinstance(rhs, Expression):
        #     return GreaterThan(lhs.name, rhs) 
        # elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
        #     # This case cannot occur. If neither is an Expression, why are you calling this method?
        #     raise Exception('What are you doing?')
        # elif not isinstance(rhs, Expression):
        #     return GreaterThan(lhs.name, rhs)
        # elif not isinstance(lhs, Expression):
        #     assert type(lhs)==str, "LHS of LessThan needs to be a string"
        #     return GreaterThan(lhs, rhs)

    def __ge__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)
        # if isinstance(lhs, Expression) and isinstance(rhs, Expression):
        #     return GreaterThan(lhs.name, rhs) 
        # elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
        #     # This case cannot occur. If neither is an Expression, why are you calling this method?
        #     raise Exception('What are you doing?')
        # elif not isinstance(rhs, Expression):
        #     return GreaterThan(lhs.name, rhs)
        # elif not isinstance(lhs, Expression):
        #     assert type(lhs)==str, "LHS of LessThan needs to be a string"
        #     return GreaterThan(lhs, rhs)

    def __eq__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of Equal needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return Equal(lhs, rhs)
        # if isinstance(lhs, Expression) and isinstance(rhs, Expression):
        #     return Equal(lhs.name, rhs)
        # elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
        #     # This case cannot occur. If neither is an Expression, why are you calling this method?
        #     raise Exception('What are you doing?')
        # elif not isinstance(rhs, Expression):
        #     return Equal(lhs.name, rhs) 
        # elif not isinstance(lhs, Expression):
        #     assert type(lhs)==str, "LHS of LessThan needs to be a string"
        #     return Equal(lhs, rhs)

    def __ne__(lhs, rhs):
        raise NotImplementedError("Not supported yet")
        
    def __str__(self):
        return str(self.name)

#     def __repr__(self):
#         return repr()