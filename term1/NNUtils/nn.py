import numpy as np


class Node:
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

        self.value = None

    def forward(self):
        """
        compute value based on inbound_nodes and store in value
        :return:
        """
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    def __init__(self, x, y):
         Node.__init__(self, [x, y])

    def forward(self):
        self.value = sum([n.value for n in self.inbound_nodes])


class Linear(Node):
    def __init__(self, features, weights, bias):
        """

        :param features: np array
        :param weights: np array
        :param bias: float
        :return:
        """
        Node.__init__(self, [features, weights, bias])

    def forward(self):
        features, weights, bias = self.inbound_nodes
        self.value = np.dot(features.value, weights.value) + bias.value


#### Utility methods #####

def topological_sort(feed_dict):
    """

    :param feed_dict: dict of input nodes and their initial values
    :return: topologically sorted list of nodes
    """
    G = {}
    nodes = list(feed_dict.keys())
    while nodes:
        node = nodes.pop(0)
        if node not in G:
            G[node] = {'in': set(), 'out': set()}
        for m in node.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[m]['in'].add(node)
            G[node]['out'].add(m)
            nodes.append(m)

    # start kahn's topsort (this version does not check for cycles as we know NN won't have any)
    nodes, sorted_nodes = list(feed_dict.keys()), []
    while nodes:
        node = nodes.pop()
        # set value of input nodes
        if isinstance(node, Input):
            node.value = feed_dict[node]
        # remove all outward edges from the node
        for m in node.outbound_nodes:
            G[node]['out'].remove(m)
            G[m]['in'].remove(node)
            if not G[m]['in']:
                nodes.append(m)
        sorted_nodes.append(node)
    return sorted_nodes

def forward_pass(output_node, sorted_nodes):
    for node in sorted_nodes:
        node.forward()
    return output_node.value
    pass


