import torch
import time
from torch import nn
import scipy.sparse as sp
import numpy as np
# from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg, diags, identity


class GF_CF(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        # A=(D^-0.5(A+sigmaI)D^-0.5)^k

    def train(self, sigma):
        adj_mat = self.adj_mat
        adj_mat = csc_matrix(adj_mat)
        dim = adj_mat.shape[0]
        start = time.time()
        adj_mat = adj_mat + sigma * identity(dim)  # augmented adj
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)  # D^-0.5
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj  # (D^-0.5(A+sigmaI)D^-0.5)
        self.d_mat = d_mat
        # self.norm_adj = norm_adj.tocsc()  # (D^-0.5(A+sigmaI)D^-0.5)
        # self.d_mat = d_mat.tocsc()
        end = time.time()
        # print('training time', end - start)

    def get_cell_position(self, k, cell_pos):
        norm_adj = self.norm_adj
        if k == 1:
            start = time.time()
            result = norm_adj @ cell_pos
            end = time.time()
            print('signal processing time', end - start)
        elif k == 2:
            start = time.time()
            result = ((norm_adj @ norm_adj) @ cell_pos)
            end = time.time()
            print('signal processing time', end - start)
        elif k == 3:
            start = time.time()
            result = ((norm_adj @ norm_adj @ norm_adj) @ cell_pos)
            end = time.time()
            print('signal processing time', end - start)
        elif k == 4:
            start = time.time()
            result = ((norm_adj @ norm_adj @ norm_adj @ norm_adj) @ cell_pos)
            end = time.time()
            # print('signal processing time', end - start)
        elif k == 6:
            start = time.time()
            result = ((norm_adj @ norm_adj @ norm_adj @ norm_adj  @ norm_adj @ norm_adj) @ cell_pos)
            end = time.time()
            # print('signal processing time', end - start)
        return result

    def ideal_low_pass_filter(self, cell_pos):
        eigenvalue, eigenvector = linalg.eigsh(self.norm_adj, k=3)
        vt = eigenvector[:, 1:]
        # result = vt @ vt.T @ cell_pos
        # return result

    def eigendecomposition(self):
        eigenvalue, eigenvector = linalg.eigsh(self.norm_adj)
        return eigenvector, eigenvalue
