from collections import namedtuple
from graphviz import Digraph
import torch
from stlcg import Expression, STL_Formula
Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_stl_graph(form, node_attr=dict(style='filled',
                                          shape='box',
                                          align='left',
                                          fontsize='12',
                                          ranksep='0.1',
                                          height='0.2',
                                          fontname="monospace"),
                         graph_attr=dict(size="12,12"), show_legend=False):
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
            color = "lightskyblue"
            if torch.is_tensor(form.value):
                color = "palegreen" if form.value.requires_grad else "lightskyblue"
            dot.node(str(id(form)), form.name, fillcolor=color)
        elif type(form) == str:
            dot.node(str(id(form)), form, fillcolor="lightskyblue")
        elif isinstance(form, STL_Formula):
            dot.node(str(id(form)), form._get_name() + "\n" + str(form), fillcolor="orange")
        elif isinstance(form, Legend):
            dot.node(str(id(form)), form.name, fillcolor=form.color, color="white")
        else:
            dot.node(str(id(form)), str(form), fillcolor="white")

        # recursive call to all the components of the formula
        if hasattr(form, '_next_function'):
            for u in form._next_function():
                dot.edge(str(id(u)), str(id(form)))
                add_nodes(u)

        if hasattr(form, '_next_legend'):
            for u in form._next_legend():
                dot.edge(str(id(u)), str(id(form)), color="white")
                add_nodes(u)

    legend_names = ["input", "variable", "formula"]
    legend_colors = ["lightskyblue", "palegreen", "orange"]
    legends = [Legend(legend_names[0], legend_colors[0])]
    for i in range(1,3):
        legends.append(Legend(legend_names[i], legend_colors[i], legends[i-1]))


    # handle multiple outputs
    if show_legend is True:
        form = (form, legends[-1])
    if isinstance(form, tuple):
        for v in form:
            add_nodes(v)
    else:
        add_nodes(form)
    resize_graph(dot)

    return dot

class Legend:
    def __init__(self, name, color, next=None):
        self.name = name
        self.color = color
        self.next = next

    def _next_legend(self):
        if self.next is None:
            return []
        return [self.next]





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