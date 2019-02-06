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

def bump_transform(oper, input_tensor, mask, scale=1, large_num = LARGE_NUMBER):
    '''
    non-masked numbers will be a large positive (or negative) number, so we can take the min (or max) properly
    '''
    sign = 1 if oper == "min" else -1
    return (large_num * sign * (1 - mask) * torch.tanh(input_tensor*scale) + mask) * input_tensor

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