import cranet
from cranet import nn

import graphviz


class ModulePlotVisitor:
    def visit(self, m: nn.Module):
        pass


def form(name, type, inp, out):
    return f'{name}: {type} | {{{{input:| {inp}}}|{{output:| {out}}}}}'


class GraphVisitor:
    def __init__(self):
        self.g = graphviz.Digraph(node_attr={'shape': 'record'})

    def visit(self, x: cranet.Tensor):
        self.g.node(self.id, form())
        for d in x.dependencies:
            d.tensor


def plot_module(module: nn.Module, inp: cranet.Tensor):
    out = module(inp)
