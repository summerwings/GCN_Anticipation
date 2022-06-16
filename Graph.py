import numpy as np

class Graph():
    '''The class to generate the graph of the given joints'''
    def __init__(self,
                 max_hop=1,
                 strategy='uniform',
                 dilation=1,
                 mode = 'Simple'):
        self.max_hop = max_hop
        self.dilation = dilation

        self.mode = mode

        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)



    def get_edge(self):
        if self.mode == 'Prior':
            self.num_node = 8
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 2), (1, 4), (1, 5), (1, 6),
                             (1, 7), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7)]
            self.edge = self_link + neighbor_link
            self.center = 1


        if self.mode == 'Full':
            self.num_node = 8
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(i, j) for i in range(self.num_node) for j in range(self.num_node)]
            self.edge = neighbor_link # self_link
            self.center = 1

        if self.mode == 's0':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 3) and (j == 0 or j == 1 or j == 3):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's1':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 3) and (j == 0 or j == 3):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's2':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1) and (j == 0 or j == 1):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's3':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0) and (j == 0):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's4':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 7) and (j == 0 or j == 1 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's5':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 6) and (j == 0 or j == 1 or j == 6):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's6':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 6) and (j == 0 or j == 6):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's7':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 2) and (j == 0 or j == 1 or j == 2):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's8':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 2) and (j == 0 or j == 2):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's9':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 5) and (j == 0 or j == 5):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's10':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 5) and (j == 0 or j == 1 or j == 5):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's11':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 4) and (j == 0 or j == 4):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's12':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 7) and (j == 0 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's13':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 4) and (j == 0 or j == 1 or j == 4):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's14':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 3 or i == 6) and (j == 0 or j == 3 or j == 6):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's15':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 6 or i == 7) and (j == 0 or j == 1 or j == 6 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's16':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 6 or i == 7) and (j == 0 or j == 6 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's17':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 2 or i == 6) and (j == 0 or j == 2 or j == 6):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's18':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 2 or i == 6 or i == 7) and (j == 0 or j == 2 or j == 6 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's19':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 2 or i == 6) and (j == 0 or j == 1 or j == 2 or j == 6):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's20':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 2 or i == 7) and (j == 0 or j == 1 or j == 2 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's21':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 2 or i == 7) and (j == 0 or j == 2 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's22':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 3 or i == 6) and (j == 0 or j == 1 or j == 3 or j == 6):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's23':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 1 or i == 3 or i == 7) and (j == 0 or j == 1 or j == 3 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's24':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 3 or i == 7) and (j == 0 or j == 3 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 's25':
            self.num_node = 8

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if (i == 0 or i == 4 or i == 6 or i == 7) and (j == 0 or j == 4 or j == 6 or j == 7):
                        neighbor_link.append((i, j))

            self.edge = neighbor_link
            self.center = 1

        if self.mode == 'Simple':
            self.num_node = 8
            self_link = [(i, i) for i in range(self.num_node)]

            neighbor_link = []
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if i !=j and (i == 0):
                        neighbor_link.append((i, j))

            self.edge = self_link + neighbor_link
            self.center = 1



    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)

        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


    def normalize_digraph(self,A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


    def normalize_undigraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD
