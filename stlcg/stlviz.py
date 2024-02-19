from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable
from stlcg import Expression, STL_Formula
import IPython
Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_stl_graph(form, node_attr=dict(style='filled',
                                          shape='box',
                                          align='left',
                                          fontsize='12',
                                          ranksep='0.1',
                                          height='0.2'),
                         graph_attr=dict(size="12,12")):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """

    # node_attr = dict(style='filled',
    #                  shape='box',
    #                  align='left',
    #                  fontsize='12',
    #                  ranksep='0.1',
    #                  height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=graph_attr)
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def tensor_to_str(tensor):
        device = tensor.device.type
        req_grad = tensor.requires_grad
        if req_grad == False:
            return "input"
        tensor = tensor.detach()
        if device == "cuda":
            tensor = tensor.cpu()
        return str(tensor.numpy())

    def add_nodes(form):
        # green are optimization variables
        # blue are non-optimization variables
        # orange are formula nodes
        # red is ambiguous, could be an optimization variable or it could not.

        if torch.is_tensor(form):
            color = "palegreen" if form.requires_grad else "lightskyblue"
            dot.node(str(id(form)), tensor_to_str(form), fillcolor=color)
        elif isinstance(form, Expression):
            color = "palegreen" if form.value.requires_grad else "lightskyblue"
            dot.node(str(id(form)), form.name, fillcolor=color)
        elif type(form) == str:
            dot.node(str(id(form)), form, fillcolor="lightcoral")
        elif isinstance(form, STL_Formula):
            dot.node(str(id(form)), form._get_name() + "\n" + str(form), fillcolor="orange")
        else:
            dot.node(str(id(form)), str(form), fillcolor="lightcoral")

        # recursive call to all the components of the formula
        if hasattr(form, '_next_function'):
            for u in form._next_function():
                dot.edge(str(id(u)), str(id(form)))
                add_nodes(u)

    # def add_nodes(var):
    #     if var not in seen:
    #         if torch.is_tensor(var):
    #             # note: this used to show .saved_tensors in pytorch0.2, but stopped
    #             # working as it was moved to ATen and Variable-Tensor merged
    #             dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
    #         elif hasattr(var, 'variable'):
    #             u = var.variable
    #             name = param_map[id(u)] if params is not None else ''
    #             node_name = '%s\n %s' % (name, size_to_str(u.size()))
    #             dot.node(str(id(var)), node_name, fillcolor='lightblue')
    #         elif var in output_nodes:
    #             dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
    #         else:
    #             dot.node(str(id(var)), str(type(var).__name__))
    #         seen.add(var)
    #         if hasattr(var, 'next_functions'):
    #             for u in var.next_functions:
    #                 if u[0] is not None:
    #                     dot.edge(str(id(u[0])), str(id(var)))
    #                     add_nodes(u[0])
    #         if hasattr(var, 'saved_tensors'):
    #             for t in var.saved_tensors:
    #                 dot.edge(str(id(t)), str(id(var)))
    #                 add_nodes(t)

    
    # handle multiple outputs
    if isinstance(form, tuple):
        for v in form:
            add_nodes(v)
    else:
        add_nodes(form)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def save_graph(dot, filename, format='pdf', cleanup=True):
    dot.render(filename=filename, format=format, cleanup=cleanup)