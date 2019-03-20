import torch
import numpy as np
from abc import ABC, abstractmethod
from scripts.util import *
import IPython
# Assume inputs are already reversed.

LARGE_NUMBER = 1E4

class Maxish(torch.nn.Module):
    def __init__(self, name="Maxish input"):
        super(Maxish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1):
        '''
        The default is
         x is of size [batch_size, max_dim, x_dim]
         if scale <= 0, then the true max is used, otherwise, the softmax is used.
         '''
        if scale > 0:
            return (torch.softmax(x*scale, dim=dim)*x).sum(dim, keepdim=True)
        else:
            return x.max(dim, keepdim=True)[0]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [str(self.input_name)]

class Minish(torch.nn.Module):
    def __init__(self, name="Minish input"):
        super(Minish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1):
        '''
        The default is
         x is of size [batch_size, max_dim, ...]
         if scale <= 0, then the true min is used, otherwise, the softmin is used.
        '''
        if scale > 0:
            return (torch.softmax(-x*scale, dim=dim)*x).sum(dim, keepdim=True)
        else:
            return x.min(dim, keepdim=True)[0]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [str(self.input_name)]

class STL_Formula(torch.nn.Module):

    def __init__(self):
        super(STL_Formula, self).__init__()

    def eval_trace(self, trace, scale):
        '''
        Trace should have size [batch_size, time_dim, x_dim]
        The output eval_trace has size [batch_size, time_dim, x_dim]
        The values in eval_trace are 0 or 1 (False or True)
        '''
        raise NotImplementedError("eval_trace not yet implemented")

    def robustness_trace(self, trace, scale):
        '''
        Trace should have size [batch_size, time_dim, x_dim]
        The output robustness_trace has size [batch_size, time_dim, x_dim]
        The values in eval_trace are real numbers
        '''
        raise NotImplementedError("robustness_trace not yet implemented")

    def eval(self, trace, scale=0, time=-1):
        '''
        Trace should have size [batch_size, time_dim, x_dim]
        Extracts the eval_trace value at the given time. (Default: time=0)
        '''
        return self.eval_trace(trace, scale)[:,time,:].unsqueeze(1)

    def robustness(self, trace, scale=0, time=-1):
        '''
        Trace should have size [batch_size, time_dim, x_dim]
        Extracts the robustness_trace value at the given time. (Default: time=0)
        '''
        return self.robustness_trace(trace, scale)[:,time,:].unsqueeze(1)

    def forward(self):
        raise NotImplementedError("forward not yet implemented")

    def __str__(self):
        raise NotImplementedError("__str__ not yet implemented")


class Temporal_Operator(STL_Formula):
    def __init__(self, subformula="Temporal input", interval=None):
        super(Temporal_Operator, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if self.interval is None else self.interval
        self.rnn_dim = 1 if not self.interval else self.interval[-1]
        self.steps = 1 if not self.interval else self.interval[-1] - self.interval[0] + 1
        self.operation = None


    def _initialize_rnn_cell(self, x):
        '''
        x is [batch_size, time_dim, x_dim]
        initial rnn state is [batch_size, rnn_dim, x_dim]
        '''
        raise NotImplementedError("_initialize_rnn_cell is not implemented")

    def _rnn_cell(self, x, h0, scale, large_num=1E4):
        '''
        x is [batch_size, 1, x_dim]
        h0 is [batch_size, rnn_dim, x_dim]
        '''
        if self.operation is None:
            raise Exception()

        if self.interval is None:
            input_ = torch.cat([h0, x], dim=1)                          # [batch_size, rnn_dim+1, x_dim]
            output = state = self.operation(input_, scale, dim=1)       # [batch_size, 1, x_dim]
        else:
            h0x = torch.cat([h0, x], dim=1)                             # [batch_size, rnn_dim+1, x_dim]
            input_ = h0x[:,:self.steps,:]                               # [batch_size, self.steps, x_dim]
            output = self.operation(input_, scale, dim=1)               # [batch_size, 1, x_dim]
            state = h0x[:,1:,:]                                         # [batch_size, rnn_dim, x_dim]


        return output, state

    def _run_cell(self, x, scale):
        outputs = []
        states = []
        h = self._initialize_rnn_cell(x)                                # [batch_size, rnn_dim, x_dim]
        xs = torch.split(x, 1, dim=1)                                   # time_dim tuple
        time_dim = len(xs)
        for i in range(time_dim):
            o, h = self._rnn_cell(xs[i], h, scale)
            outputs.append(o)
            states.append(h)

        return outputs, states

    def robustness_trace(self, x, scale=0):
        outputs, states = self._run_cell(x, scale)
        return torch.cat(outputs, dim=1)                              # [batch_size, time_dim, x_dim]

    def eval_trace(self, x, scale=0):
        outputs, states = self._run_cell(x, scale)
        return torch.cat(outputs, dim=1) > 0                          # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula]

    def forward(self, x, scale=0):
        return self.robustness_trace(x, scale)


class Always(Temporal_Operator):
    def __init__(self, subformula='Always input', interval=None):
        super(Always, self).__init__(subformula=subformula, interval=interval)
        self.operation = Minish()
        self.oper = "min"

    def _initialize_rnn_cell(self, x):
        '''
        x is [batch_size, time_dim, x_dim]
        initial rnn state is [batch_size, rnn_dim, x_dim]
        '''
        init_val = LARGE_NUMBER
        h0 = torch.ones([x.shape[0], self.rnn_dim, x.shape[2]])*init_val
        return h0.to(x.device)

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"

class Eventually(Temporal_Operator):
    def __init__(self, subformula='Eventually input', interval=None):
        super(Eventually, self).__init__(subformula=subformula, interval=interval)
        self.operation = Maxish()
        self.oper = "max"

    def _initialize_rnn_cell(self, x):
        '''
        x is [batch_size, time_dim, x_dim]
        initial rnn state is [batch_size, rnn_dim, x_dim]
        '''
        init_val = -LARGE_NUMBER
        h0 = torch.ones([x.shape[0], self.rnn_dim, x.shape[2]])*init_val
        return h0.to(x.device)

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"

class LessThan(STL_Formula):
    '''
    s <= c where s is the signal, and c is the constant.
    '''
    def __init__(self, name='x', c='c'):
        super(LessThan, self).__init__()
        self.name = name
        self.c = c

    def robustness_trace(self, x, scale=1):
        if scale == 1:
            return self.c - x
        return (self.c - x)*scale

    def robustness(self, x, time=-1, scale=1):
        return self.robustness_trace(x, scale)[:,time,:].unsqueeze(1)

    def eval_trace(self, x, scale=1):
        return self.robustness_trace(x, scale) > 0

    def eval(self, x, time=-1, scale=1):
        return self.eval_trace(x, scale)[:,time,:].unsqueeze(1)

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.name, self.c]

    def forward(self, x, scale=1):
        return self.robustness_trace(x, scale)

    def __str__(self):
        return self.name + " <= " + tensor_to_str(self.c)


class GreaterThan(STL_Formula):
    '''
    s >= c where s is the signal, and c is the constant.
    '''
    def __init__(self, name='x', c='c'):
        super(GreaterThan, self).__init__()
        self.name = name
        self.c = c

    def robustness_trace(self, x, scale=1):
        if scale == 1:
            return x - self.c
        return (x - self.c)*scale

    def robustness(self, x, time=-1, scale=1):
        return self.robustness_trace(x, scale)[:,time,:].unsqueeze(1)

    def eval_trace(self, x, scale=1):
        return self.robustness_trace(x, scale) > 0

    def eval(self, x, time=-1, scale=1):
        return self.eval_trace(x, scale)[:,time,:].unsqueeze(1)

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.name, self.c]

    def forward(self, x, scale=1):
        return self.robustness_trace(x, scale)

    def __str__(self):
        return self.name + " >= " + tensor_to_str(self.c)

class Equal(STL_Formula):
    '''
    s = c where s is the signal, and c is the constant.
    '''
    def __init__(self, name='x', c='c'):
        super(Equal, self).__init__()
        self.name = name
        self.c = c

    def robustness_trace(self, x, scale=1):
        if scale == 1:
            return  torch.abs(x - self.c)
        return torch.abs(x - self.c)*scale

    def robustness(self, x, time=-1, scale=1):
        return self.robustness_trace(x, scale)[:,time,:].unsqueeze(1)

    def eval_trace(self, x, scale=1):
        return self.robustness_trace(x, scale) > 0

    def eval(self, x, time=-1, scale=1):
        return self.eval_trace(x, scale)[:,time,:].unsqueeze(1)

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.name, self.c]

    def forward(self, x, scale=1):
        return self.robustness_trace(x, scale)

    def __str__(self):
        return self.name + " = " + tensor_to_str(self.c)

class Negation(STL_Formula):
    '''
    not Subformula
    '''
    def __init__(self, subformula='Negation input'):
        super(Negation, self).__init__()
        self.subformula = subformula

    def robustness_trace(self, x, scale=1):
        if scale == 1:
            return -x
        return -x*scale

    def robustness(self, x, time=-1, scale=1):
        return self.robustness_trace(x, scale)[:,time,:].unsqueeze(1)

    def eval_trace(self, x, scale=1):
        return -x*scale > 0

    def eval(self, x, time=-1, scale=1):
        return self.eval_trace(x, scale)[:,time,:].unsqueeze(1)

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula]

    def forward(self, x, scale=1):
        return self.robustness_trace(x, scale)

    def __str__(self):
        return "¬(" + str(self.subformula) + ")"

class And(STL_Formula):
    def __init__(self, subformula1="And subformula1", subformula2="And subformula2"):
        super(And, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Minish()

    def robustness_trace(self, trace1, trace2, scale=0):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        xx = torch.stack([trace1, trace2], dim=-1)
        return self.operation(xx, scale, dim=-1).squeeze(-1)                                         # [batch_size, time_dim, x_dim]

    def robustness(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness_trace(trace1, trace2, scale)[:,time,:].unsqueeze(1)           # [batch_size, time_dim, x_dim]

    def eval_trace(self, trace1, trace2, scale=0):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        xx = torch.stack([trace1, trace2], dim=-1)
        return self.operation(xx, scale, dim=-1) > 0                                                 # [batch_size, time_dim, x_dim]

    def eval(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.eval_trace(trace1, trace2, scale)[:,time,:].unsqueeze(1)                 # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def forward(self, trace1, trace2, scale=0):
        return self.robustness_trace(trace1, trace2, scale)

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"

class Or(STL_Formula):
    def __init__(self, subformula1="Or subformula1", subformula2="Or subformula2"):
        super(Or, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Maxish()

    def robustness_trace(self, trace1, trace2, scale=0):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        xx = torch.stack([trace1, trace2], dim=-1)
        return self.operation(xx, scale, dim=-1).squeeze(-1)                                         # [batch_size, time_dim, x_dim]

    def robustness(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness_trace(trace1, trace2, scale)[:,time,:].unsqueeze(1)           # [batch_size, time_dim, x_dim]

    def eval_trace(self, trace1, trace2, scale=0):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        xx = torch.stack([trace1, trace2], dim=-1)
        return self.operation(xx, scale, dim=-1) > 0                                                 # [batch_size, time_dim, x_dim]

    def eval(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.eval_trace(trace1, trace2, scale)[:,time,:].unsqueeze(1)                 # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def forward(self, trace1, trace2, scale=0):
        return self.robustness_trace(trace1, trace2, scale)

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"


class Until(STL_Formula):
    def __init__(self, subformula1="Until subformula1", subformula2="Until subformula2"):
        super(Until, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, trace1, trace2, scale=0):
        '''
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        Alw = Always()
        minish = Minish()
        maxish = Maxish()
        LHS = trace2.unsqueeze(-1).repeat([1, 1, 1,trace2.shape[1]]).permute(0, 3, 2, 1)                                  # [batch_size, time_dim, x_dim, time_dim]
        RHS = torch.ones(LHS.shape)*-1000000                                                    # [batch_size, time_dim, x_dim, time_dim]
        for i in range(trace2.shape[1]):
            RHS[:,i:,:,i] = Alw(trace1[:,i:,:])
        # first min over the (ρ(ψ), ◻ρ(ϕ))
        # then max over the t′ dimension (the second time_dime dimension)
        return maxish(minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1).squeeze(-1), scale=scale, dim=-1).squeeze(-1)                                                              # [batch_size, time_dim, x_dim]

    def robustness(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness_trace(trace1, trace2, scale)[:,time,:].unsqueeze(1)           # [batch_size, time_dim, x_dim]

    def eval_trace(self, trace1, trace2, scale=0):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness_trace(trace1, trace2, scale) > 0                               # [batch_size, time_dim, x_dim]

    def eval(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness(trace1, trace2, scale, time) > 0                               # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def forward(self, trace1, trace2, scale=0):
        return self.robustness_trace(trace1, trace2, scale)

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + "(" + str(self.subformula2) + ")"

class Then(STL_Formula):
    def __init__(self, subformula1="Then subformula1", subformula2="Then subformula2"):
        super(Then, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, trace1, trace2, scale=0):
        '''
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        Ev = Eventually()
        minish = Minish()
        maxish = Maxish()
        LHS = trace2.unsqueeze(-1).repeat([1, 1, 1,trace2.shape[1]]).permute(0, 3, 2, 1)                                  # [batch_size, time_dim, x_dim, time_dim]
        RHS = torch.ones(LHS.shape)*-1000000                                                    # [batch_size, time_dim, x_dim, time_dim]
        for i in range(trace2.shape[1]):
            RHS[:,i:,:,i] = Ev(trace1[:,i:,:])
        # first min over the (ρ(ψ), ◻ρ(ϕ))
        # then max over the t′ dimension (the second time_dime dimension)
        return maxish(minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1).squeeze(-1), scale=scale, dim=-1).squeeze(-1)                                                              # [batch_size, time_dim, x_dim]

    def robustness(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness_trace(trace1, trace2, scale)[:,time,:].unsqueeze(1)           # [batch_size, time_dim, x_dim]

    def eval_trace(self, trace1, trace2, scale=0):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness_trace(trace1, trace2, scale) > 0                               # [batch_size, time_dim, x_dim]

    def eval(self, trace1, trace2, scale=0, time=-1):
        '''
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        return self.robustness(trace1, trace2, scale, time) > 0                               # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def forward(self, trace1, trace2, scale=0):
        return self.robustness_trace(trace1, trace2, scale)

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " T " + "(" + str(self.subformula2) + ")"


# class STLModel(torch.nn.Module):
#     def __init__(self, inner, outer):
#         super(STLModel, self).__init__()
#         self.inner = inner
#         self.outer = outer
#     def __str__(self):
#         return str(self.outer) + " ( " + str(self.inner) + " ) "

#     def setup(self, x, y=None, scale=5):
#         return self.inner(x, y, scale=5)

#     def forward(self, x, y=None, dim=1, scale=5, cuda=False):
#         if self.outer.interval is not None:
#             if dim == 0:
#                 x = x[self.outer.interval.lower():self.outer.interval.upper()+1]
#                 if y is not None:
#                     y = y[self.outer.interval.lower():self.outer.interval.upper()+1]
#             elif dim == 1:
#                 x = x[:,self.outer.interval.lower():self.outer.interval.upper()+1]
#                 if y is not None:
#                     y = y[:,self.outer.interval.lower():self.outer.interval.upper()+1]
#             else:
#                 raise NotImplementedError("dim = ", dim, " is not implemented")
#         x_input = self.setup(x, y, scale=scale)
#         return self.outer(x_input, dim=dim, scale=scale, cuda=cuda)

# def sigmoid(x):
#     return 1/(1+np.exp(-x))


