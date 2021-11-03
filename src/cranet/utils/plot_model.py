import cranet

import graphviz


class TensorDepVisitor:
    def __init__(self):
        self.g = graphviz.Digraph(node_attr={'shape': 'record'})
        self.ids = 0

    def form(self, shape):
        return f'{shape}'

    def graph(self):
        return self.g

    def visit(self, x: cranet.Tensor):
        id = str(self.ids)
        self.ids += 1
        self.g.node(id, self.form(str(x.data.shape)))
        for d in x.dependencies:
            a_id = self.visit(d.tensor)
            self.g.edge(a_id, id, label=d.meta['name'])
        return id


def plot_tensor_dep(x: cranet.Tensor):
    gv = TensorDepVisitor()
    gv.visit(x)
    return gv.graph()
