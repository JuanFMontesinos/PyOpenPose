from torchtree import Tree as _Tree


def get_openpose_graph():
    return [(4, 3), (3, 2), (7, 6), (6, 5),
            (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
            (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
            (17, 15), (16, 14)]


def get_openpose_hands_graph():
    return [(4, 3), (3, 2), (2, 1), (1, 0),
            (8, 7), (7, 6), (6, 5), (5, 0),
            (12, 11), (11, 10), (10, 9), (9, 0),
            (16, 15), (15, 14), (14, 13), (13, 0),
            (20, 19), (19, 18), (18, 17), (17, 0)]


def get_openpose_hands_graph():
    neighbor_link_body = [(4, 3), (3, 2), (7, 6), (6, 5),
                          (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                          (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                          (17, 15), (16, 14)]
    neighbor_link_hand1 = [(4, 3), (3, 2), (2, 1), (1, -13),
                           (8, 7), (7, 6), (6, 5), (5, -13),
                           (12, 11), (11, 10), (10, 9), (9, -13),
                           (16, 15), (15, 14), (14, 13), (13, -13),
                           (20, 19), (19, 18), (18, 17), (17, -13)]
    neighbor_link_hand1 = [(x[0] + 17, x[1] + 17) for x in neighbor_link_hand1]
    neighbor_link_hand2 = [(4, 3), (3, 2), (2, 1), (1, -30),
                           (8, 7), (7, 6), (6, 5), (5, -30),
                           (12, 11), (11, 10), (10, 9), (9, -30),
                           (16, 15), (15, 14), (14, 13), (13, -30),
                           (20, 19), (19, 18), (18, 17), (17, -30)]
    neighbor_link_hand2 = [(x[0] + 17 + 20, x[1] + 17 + 20) for x in neighbor_link_hand2]
    return neighbor_link_body + neighbor_link_hand1 + neighbor_link_hand2


def get_upperbody_with_hands_graph():
    neighbor_link_body = [(3, 2), (2, 1), (1, 0),
                          (6, 5), (5, 4), (4, 0)]
    neighbor_link_hand1 = [(4, 3), (3, 2), (2, 1), (1, -3),
                           (8, 7), (7, 6), (6, 5), (5, -3),
                           (12, 11), (11, 10), (10, 9), (9, -3),
                           (16, 15), (15, 14), (14, 13), (13, -3),
                           (20, 19), (19, 18), (18, 17), (17, -3)]
    neighbor_link_hand1 = [(x[0] + 6, x[1] + 6) for x in neighbor_link_hand1]
    neighbor_link_hand2 = [(4, 3), (3, 2), (2, 1), (1, -20),
                           (8, 7), (7, 6), (6, 5), (5, -20),
                           (12, 11), (11, 10), (10, 9), (9, -20),
                           (16, 15), (15, 14), (14, 13), (13, -20),
                           (20, 19), (19, 18), (18, 17), (17, -20)]
    neighbor_link_hand2 = [(x[0] + 6 + 20, x[1] + 6 + 20) for x in neighbor_link_hand2]
    return neighbor_link_body + neighbor_link_hand1 + neighbor_link_hand2


def get_upperbody_graph():
    return [(3, 2), (2, 1), (1, 0),
            (6, 5), (5, 4), (4, 0)]


class Graph(object):
    def __init__(self, graph):
        self.graph = sorted([sorted(x) for x in graph])
        self.tree = _Tree()
        self.build_tree(self.tree, 0)

    def iter_graph(self, c):
        triggered = False
        for edge in self.graph:
            if edge[0] == c and edge[1] != c:
                triggered = True
                yield edge
        if not triggered:
            yield (c, None)

    def build_tree(self, tree, c):
        for edge in self.iter_graph(c):
            x, y = str(edge[0]), edge[1]
            if y is None:
                tree.add_module(x, _Tree())
            else:
                if x not in tree._modules and int(x) != y:

                    new_tree = _Tree()
                    tree.add_module(x, new_tree)
                    self.build_tree(new_tree, y)
                else:
                    self.build_tree(new_tree, y)

    def get_ordered_graph(self):
        x = []
        self.iter_ordered_graph(self.tree, -1, x)
        del x[0]
        return tuple(x)

    def iter_ordered_graph(self, tree, prev, lista):
        for el, mod in self.iter_children(tree, prev):
            lista.append(el)
            self.iter_ordered_graph(mod, el[1], lista)

    def iter_children(self, tree, prev):
        for name, mod in tree.named_children():
            yield (prev, int(name)), mod
