# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import sys

# from utils import tensor_to_str
'''
Important information:
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
        assert x_._value is not None, "Input Expression does not have numerical values"
        x_ret = x_._value
    elif isinstance(x_, torch.Tensor):
        x_ret = x_
    elif isinstance(x_, tuple):
        x_ret = convert_to_input_values(x_)
    else:
        raise ValueError("First argument is an invalid input trace")

    if isinstance(y_, Expression):
        assert y_._value is not None, "Input Expression does not have numerical values"
        y_ret = y_._value
    elif isinstance(y_, torch.Tensor):
        y_ret = y_
    elif isinstance(y_, tuple):
        y_ret = convert_to_input_values(y_)
    else:
        raise ValueError("Second argument is an invalid input trace")

    return (x_ret, y_ret)
    

class Maxish(torch.nn.Module):
    def __init__(self, name="Maxish input"):
        super(Maxish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1, keepdim=True):
        '''
        The default is
         x is of size [batch_size, N, ...] where N is typically the trace length
         if scale <= 0, then the true max is used, otherwise, the softmax is used.
        '''
        if isinstance(x, Expression):
            assert x._value is not None, "Input Expression does not have numerical values"
            x = x._value
        if scale > 0:
            return (torch.softmax(x*scale, dim=dim)*x).sum(dim, keepdim=keepdim)
        else:
            return x.max(dim, keepdim=keepdim)[0]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [str(self.input_name)]

class Minish(torch.nn.Module):
    def __init__(self, name="Minish input"):
        super(Minish, self).__init__()
        self.input_name = name

    def forward(self, x, scale, dim=1, keepdim=True):
        '''
        The default is
         x is of size [batch_size, N, ...] where N is typically the trace length
         if scale <= 0, then the true min is used, otherwise, the softmin is used.
        '''
        if isinstance(x, Expression):
            assert x._value is not None, "Input Expression does not have numerical values"
            x = x._value

        if scale > 0:
            return (torch.softmax(-x*scale, dim=dim)*x).sum(dim, keepdim=keepdim)
        else:
            return x.min(dim, keepdim=keepdim)[0]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [str(self.input_name)]

class STL_Formula(torch.nn.Module):

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, trace, pscale=1, scale=-1, **kwargs):
        '''
        Trace should have size [batch_size, time_dim, ...]
        The output robustness_trace has size [batch_size, time_dim, ...]
        The values in eval_trace are real numbers
        '''
        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, inputs, time=0, pscale=1, scale=-1, **kwargs):
        '''
        Trace should have size [batch_size, time_dim, x_dim]
        Extracts the robustness_trace value at the given time. (Default: time=0 assuming reverse trace)
        '''
        return self.robustness_trace(inputs, pscale=pscale, scale=scale)[:,-(time+1),:].unsqueeze(1)

    def eval_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        '''
        Trace should have size [batch_size, time_dim, x_dim]
        The output eval_trace has size [batch_size, time_dim, x_dim]
        The values in eval_trace are 0 or 1 (False or True)
        '''
        return self.robustness_trace(inputs, pscale=pscale, scale=scale) > 0

    def eval(self, inputs, time=0, pscale=1, scale=-1, **kwargs):
        '''
        Trace should have size [batch_size, time_dim, x_dim]
        Extracts the eval_trace value at the given time. (Default: time=0)
        '''
        return self.eval_trace(inputs, pscale=pscale, scale=scale)[:,-(time+1),:].unsqueeze(1)                 # [batch_size, time_dim, x_dim]

    def forward(formula, inputs, pscale=1, scale=-1, **kwargs):
        '''
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.
        '''
        if isinstance(inputs, Expression):
            assert inputs._value is not None, "Input Expression does not have numerical values"
            return formula.robustness_trace(inputs._value, pscale=pscale, scale=scale, **kwargs)
        elif isinstance(inputs, torch.Tensor):
            return formula.robustness_trace(inputs, pscale=pscale, scale=scale, **kwargs)
        elif isinstance(inputs, tuple):
            return formula.robustness_trace(convert_to_input_values(inputs), pscale=pscale, scale=scale, **kwargs)
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

    def robustness_trace(self, trace, pscale=1, **kwargs):
        return trace * pscale 

    def _next_function(self):
        return []

    def __str__(self):
        return "%s" %self.name




class Temporal_Operator(STL_Formula):
    def __init__(self, subformula, interval=None):
        super(Temporal_Operator, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if self.interval is None else self.interval
        self.rnn_dim = 1 if not self.interval else self.interval[-1]    # rnn_dim=1 if interval is [0, ∞) otherwise rnn_dim=end of interval
        self.steps = 1 if not self.interval else self.interval[-1] - self.interval[0] + 1   # steps=1 if interval is [0, ∞) otherwise steps=length of interval
        self.operation = None


    def _initialize_rnn_cell(self, x):
        '''
        x is [batch_size, time_dim, ...]
        initial rnn state is [batch_size, rnn_dim, ...]
        '''
        raise NotImplementedError("_initialize_rnn_cell is not implemented")

    def _rnn_cell(self, x, h0, scale=-1, **kwargs):
        '''
        x: rnn input [batch_size, 1, ...]
        h0: input rnn hidden state  [batch_size, rnn_dim, ...]
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

    def _run_rnn_cell(self, x, scale):
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

    def robustness_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        trace = self.subformula(inputs, pscale=pscale, scale=scale)
        outputs, states = self._run_rnn_cell(trace, scale=scale)
        return torch.cat(outputs, dim=1)                              # [batch_size, time_dim, ...]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula]



class Always(Temporal_Operator):
    def __init__(self, subformula, interval=None):
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
    x <= c where x is the signal, and c is the constant.
    '''
    def __init__(self, name='x', val='c'):
        super(LessThan, self).__init__()
        assert isinstance(name, str), "LHS of expression needs to be a string (input name)"
        self.input_name = name
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, pscale=1, **kwargs):
        if pscale == 1:
            return self.val - trace
        return (self.val - trace)*pscale

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.input_name, self.val]

    def __str__(self):
        return self.input_name + " <= " + tensor_to_str(self.val)


class GreaterThan(STL_Formula):
    '''
    x >= c where x is the signal, and c is the constant.
    '''
    def __init__(self, name='x', val='c'):
        super(GreaterThan, self).__init__()
        assert isinstance(name, str), "LHS of expression needs to be a string (input name)"
        self.input_name = name
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, pscale=1, **kwargs):
        if pscale == 1:
            return trace - self.val
        return (trace - self.val)*pscale

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.input_name, self.val]

    def __str__(self):
        return self.input_name + " >= " + tensor_to_str(self.val)

class Equal(STL_Formula):
    '''
    s = c where s is the signal, and c is the constant.
    '''
    def __init__(self, name='x', val='c'):
        super(Equal, self).__init__()
        assert isinstance(name, str), "LHS of expression needs to be a string (input name)"
        self.input_name = name
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, pscale=1, **kwargs):
        if pscale == 1:
            return  torch.abs(trace - self.val)
        return torch.abs(trace - self.val)*pscale

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.input_name, self.val]

    def __str__(self):
        return self.input_name + " = " + tensor_to_str(self.val)

class Negation(STL_Formula):
    '''
    not Subformula
    '''
    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

    def robustness_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        return -self.subformula(inputs, pscale=pscale, scale=scale)*pscale

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


    def robustness_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale)
        xx = torch.stack([-trace1, trace2], dim=-1)      # [batch_size, time_dim, ..., 2]
        return self.operation(xx, scale, dim=-1, keepdim=False)   # [batch_size, time_dim, ...]

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

    def robustness_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale)
        xx = torch.stack([trace1, trace2], dim=-1)      # [batch_size, time_dim, ..., 2]
        return self.operation(xx, scale, dim=-1, keepdim=False)                                         # [batch_size, time_dim, ...]

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

    def robustness_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale)
        xx = torch.stack([trace1, trace2], dim=-1)      # [batch_size, time_dim, ..., 2]
        return self.operation(xx, scale, dim=-1, keepdim=False)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"


class Until(STL_Formula):
    def __init__(self, subformula1="Until subformula1", subformula2="Until subformula2"):
        '''
        subformula1 U subformula2
        '''
        super(Until, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        '''
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale)
        Alw = Always(subformula=Identity(name=str(self.subformula1)))
        minish = Minish()
        maxish = Maxish()
        LHS = trace2.unsqueeze(-1).repeat([1, 1, 1,trace2.shape[1]]).permute(0, 3, 2, 1)                                  # [batch_size, time_dim, x_dim, time_dim]
        RHS = torch.ones(LHS.shape)*-LARGE_NUMBER                                                    # [batch_size, time_dim, x_dim, time_dim]
        for i in range(trace2.shape[1]):
            RHS[:,i:,:,i] = Alw(trace1[:,i:,:], pscale=pscale, scale=scale)
        # first min over the (ρ(ψ), ◻ρ(ϕ))
        # then max over the t′ dimension (the second time_dime dimension)
        return maxish(minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1).squeeze(-1), scale=scale, dim=-1).squeeze(-1)                                                              # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + "(" + str(self.subformula2) + ")"

class Then(STL_Formula):
    '''
    subformula1 T subformula2
    '''
    def __init__(self, subformula1, subformula2):
        super(Then, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs, pscale=1, scale=-1, **kwargs):
        '''
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        '''
        x, y = inputs
        trace1 = self.subformula1(x, pscale=pscale, scale=scale)
        trace2 = self.subformula2(y, pscale=pscale, scale=scale)
        Ev = Eventually(subformula=Identity(name=str(self.subformula1)))
        minish = Minish()
        maxish = Maxish()
        LHS = trace2.unsqueeze(-1).repeat([1, 1, 1,trace2.shape[1]]).permute(0, 3, 2, 1)                                  # [batch_size, time_dim, x_dim, time_dim]
        RHS = torch.ones(LHS.shape)*-LARGE_NUMBER                                                 # [batch_size, time_dim, x_dim, time_dim]
        for i in range(trace2.shape[1]):
            RHS[:,i:,:,i] = Ev(trace1[:,i:,:], pscale=pscale, scale=scale)
        # first min over the (ρ(ψ), ◻ρ(ϕ))
        # then max over the t′ dimension (the second time_dime dimension)
        return maxish(minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1).squeeze(-1), scale=scale, dim=-1).squeeze(-1)                                                              # [batch_size, time_dim, x_dim]

    def _next_function(self):
        # next function is actually input (traverses the graph backwards)
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " T " + "(" + str(self.subformula2) + ")"


class Expression(torch.nn.Module):
    '''
    Wraps a pytorch arithmetic operation, so that we can intercept and overload comparison operators.
    '''
    def __init__(self, value, num_val=None):
        super(Expression, self).__init__()
        self.value = value
        self._value = num_val
    def __neg__(self):
        return Expression(-self.value)
    
    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.value + other.value)
        else:
            return Expression(self.value + other)

    def __radd__(self, other):
        return self.__add__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular add 

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(self.value - other.value)
        else:
            return Expression(self.value - other)

    def __rsub__(self, other):
        return Expression(other - self.value)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(self.value * other.value)
        else:
            return Expression(self.value * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        # This is the new form required by Python 3
        numerator = a
        denominator = b
        if isinstance(numerator, Expression):
            numerator = numerator.value
        if isinstance(denominator, Expression):
            denominator = denominator.value

        return Expression(numerator/denominator)
        
    # Comparators
    def __lt__(lhs, rhs):
        if isinstance(lhs, Expression) and isinstance(rhs, Expression):
            return LessThan(lhs.value, rhs.value) 
        elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
            # This case cannot occur. If neither is an Expression, why are you calling this method?
            raise Exception('What are you doing?')
        elif not isinstance(rhs, Expression):
            return LessThan(lhs.value, rhs)
        elif not isinstance(lhs, Expression):
            return LessThan(lhs, rhs.value)

    def __le__(lhs, rhs):
        if isinstance(lhs, Expression) and isinstance(rhs, Expression):
            return LessThan(lhs.value, rhs.value) 
        elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
            # This case cannot occur. If neither is an Expression, why are you calling this method?
            raise Exception('What are you doing?')
        elif not isinstance(rhs, Expression):
            return LessThan(lhs.value, rhs)
        elif not isinstance(lhs, Expression):
            return LessThan(lhs, rhs.value)

    def __gt__(lhs, rhs):
        if isinstance(lhs, Expression) and isinstance(rhs, Expression):
            return GreaterThan(lhs.value, rhs.value) 
        elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
            # This case cannot occur. If neither is an Expression, why are you calling this method?
            raise Exception('What are you doing?')
        elif not isinstance(rhs, Expression):
            return GreaterThan(lhs.value, rhs)
        elif not isinstance(lhs, Expression):
            return GreaterThan(lhs, rhs.value)

    def __ge__(lhs, rhs):
        if isinstance(lhs, Expression) and isinstance(rhs, Expression):
            return GreaterThan(lhs.value, rhs.value) 
        elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
            # This case cannot occur. If neither is an Expression, why are you calling this method?
            raise Exception('What are you doing?')
        elif not isinstance(rhs, Expression):
            return GreaterThan(lhs.value, rhs)
        elif not isinstance(lhs, Expression):
            return GreaterThan(lhs, rhs.value)

    def __eq__(lhs, rhs):
        if isinstance(lhs, Expression) and isinstance(rhs, Expression):
            return Equal(lhs.value, rhs.value)
        elif (not isinstance(lhs, Expression)) and (not isinstance(rhs, Expression)):
            # This case cannot occur. If neither is an Expression, why are you calling this method?
            raise Exception('What are you doing?')
        elif not isinstance(rhs, Expression):
            return Equal(lhs.value, rhs) 
        elif not isinstance(lhs, Expression):
            return Equal(lhs, rhs.value)

    def __ne__(lhs, rhs):
        raise NotImplementedError("Not supported yet")

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)