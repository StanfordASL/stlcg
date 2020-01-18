# -*- coding: utf-8 -*-
import torch
import numpy as np

LARGE_NUMBER = 1E4

def bump(input_tensor, left, right, slope):
    '''
    creating the bump function
    σ(s(x-a))(1 - σ(s(x - b)))
    '''
    mask = (torch.sigmoid(slope*(input_tensor - left))*(1 - torch.sigmoid(slope*(input_tensor - right))))
    return mask/torch.max(mask)

def bump_transform(oper, input_tensor, mask, scale=1, large_num=LARGE_NUMBER):
    '''
    non-masked numbers will be a large positive (or negative) number, so we can take the min (or max) properly
    '''
    sign = 1 if oper == "min" else -1
    sgn = torch.sign(input_tensor) if scale <= 0.0 else torch.tanh(input_tensor*scale) 
    return (large_num * sign * (1 - mask) * sgn + mask) * input_tensor

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

def print_learning_progress(formula, inputs, var_dict, i, loss, scale):
    vals = [i, loss]
    string = "iteration: %i -- loss: %.3f"
    for (k,v) in var_dict.items():
        string += " ---- %s:%.3f"
        vals.append(k)
        vals.append(v)
    string += " ---- scale:%.3f"
    vals.append(scale)
    string += " ---- true value:%.3f"
    vals.append(formula.robustness(inputs).detach().numpy())
    print(string%tuple(vals))