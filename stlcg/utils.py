import torch
from stlcg import Expression

LARGE_NUMBER = 1E4

def reverse_time(signal, dim):
    if isinstance(signal, Expression):
        signal.flip(dim)
    elif isinstance(signal, torch.Tensor):
        signal = signal.flip(dim)
    else:
        raise ValueError("Unknown signal type")
    return signal

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
    vals.append(formula.robustness(inputs).mean().detach().cpu().numpy())
    print(string%tuple(vals))


def plot_add_signal_Expression(ax, exp : Expression, time_dim=1, time=None, plot_reversed=False, **fmt):
    trace = exp.value.detach().cpu()
    if exp.reversed is not plot_reversed:
        trace = trace.flip(time_dim)
    if time is None:
        time = range(trace.shape[time_dim])
    for tr in trace:
        ax.plot(time, tr, '.-', label=exp.name, **fmt)
    return ax

