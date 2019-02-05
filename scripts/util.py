import torch
import numpy as np


def bump(input_tensor, left, right, slope):
    mask = (torch.sigmoid(slope*(input_tensor - left))*(1 - torch.sigmoid(slope*(input_tensor - right))))
    return mask/torch.max(mask)

def tensor_to_str(tensor):
    device = tensor.device.type
    req_grad = tensor.requires_grad
    if req_grad == False:
        return "input"
    tensor = tensor.detach()
    if device == "cuda":
        tensor = tensor.cpu()
    return str(tensor.numpy())