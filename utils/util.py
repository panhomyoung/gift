import re
import time
import json
import csv
import time
import numpy as np
import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np
import statistics
from scipy import stats
from pathlib import Path
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg


def make_dir(path):
    if os.path.isdir(path):
        print(path, ' dir exists')
    else:
        os.makedirs(path)
        print(path, 'is created')


def save_dict_as_json(dictionary: dict, savepath: Path):
    """ save dictionary as json file
    :param dictionary: the source
    :param savepath: the directory of target file
    :return: None
    """
    # create path if not exist
    savepath.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(dictionary)
    with open(savepath, 'w') as ff:
        ff.write(json_str)


def read_cluster_result(file: Path) -> list:
    """
    load cluster results as list, where the list index corresponding to the sparse matrix index
    :param file: the directory of cluster result file
    :return: cluster result list
    """
    with open(file, 'r')as f:
        content = csv.reader(f)
        cluster_result = []
        for i in content:
            for x in i:
                cluster_result.append(int(x))
    return cluster_result


def cal_adj_matrix(feature):
    adj_matrix = np.zeros(shape=(len(feature), len(feature[0])), dtype="double")
    connectivity_index = np.where(feature > 0)  # connectivity_index is a tuple data, each element is an array
    for i, j in zip(connectivity_index[0], connectivity_index[1]):
        adj_matrix[i][j] = 1
    return adj_matrix


class Parser:
    def __init__(self):
        self.nodeName2MatrixIndex = dict()
        self._terminalNumber = 0
        self._nodesNumber = -1
        self._netNumber = 0
        self._pinNumber = 0

    def extract_nodes_name(self, nodefilepath: Path) -> list:
        """
        extract all nodes' name as a list
        :param nodefilepath: the directory of node file
        :return: a list consists name of all nodes
        """
        with open(nodefilepath, 'r')as f:
            info = f.read()
            nodeNumberRegex = re.compile(r'NumNodes\s*:\s*(\d+)\s+')
            terminalNumberRegex = re.compile(r'NumTerminals\s*:\s*(\d+)\s+')
            self._nodesNumber = int(re.findall(nodeNumberRegex, info)[0])
            self._terminalNumber = int(re.findall(terminalNumberRegex, info)[0])

            nodeRegex = re.compile(r'\s+(o\d+)\s+\d+\s+\d+')
            node_info = re.findall(nodeRegex, info)
        return node_info

    def construct_nodeName2MatrixIndex_dictionary(self, node_name: list, nodeName2MatrixIndex: Path):
        """ construct a ||| cell_name : sparse_matrix_index ||| dictionary
        :param node_name: a list contains all the nodes name
        :param nodeName2MatrixIndex: the file directory of the cellName2MatrixIndex
        :return:
            None, save the dictionary to specific path
        """
        # for i in range(len(node_name)):
        #     self.nodeName2MatrixIndex.setdefault(node_name[i], i)
        for i, node in enumerate(node_name):
            self.nodeName2MatrixIndex.setdefault(node, i)
        if len(self.nodeName2MatrixIndex) == self._nodesNumber:
            print("cell number = matrix index it's ok to save cellName2MatrixIndex")
            save_dict_as_json(self.nodeName2MatrixIndex, nodeName2MatrixIndex)
        else:
            print("cell number != matrix index, check the nodes file")
            print("exiting the program ...")
            exit()

    def extract_net_info(self, net_file_path: Path) -> list:
        """
        extract net information as list where each element in list is a subnet
        :param net_file_path: the directory of net file
        :return: subnets info list, each element is a string of subnets info
        """
        a = time.time()
        with open(net_file_path, 'r') as f:
            netlist = f.read()
            self._netNumber = int(re.findall(r'NumNets\s*:\s*(\d+)', netlist)[0])
            self._pinNumber = int(re.findall(r'NumPins\s*:\s*(\d+)', netlist)[0])
            subnetRegex = re.compile(r'NetDegree\s*:')
            subnet = re.split(subnetRegex, netlist)
            subnet.pop(0)
        b = time.time()
        print(f"extract subnets info spends {b - a}s.")
        return subnet

    def extract_nodes_in_subnets(self, subnets: list) -> tuple:
        """ extract root cell name and adjacent cell name correlating with root cell in each subnet.
        If the subnet only contains 1 cell, then it stores in root cell list and its corresponding adjacent
        cell list still exists but a [](empty list).
        :param subnets: a list of subnets in which each element is a string including all info of subnet
        :return: list of root cell name, list[list] of adjacent cell name, int: total_pins
        """

        if len(subnets) == self._netNumber:
            a = time.time()
            print(f"NumNets is right, start extracting nodes in subnets")
            allRootCell = []
            allAdjacentCell = []
            allCell = []
            adjacentcellRegex = re.compile(r'o\d+')
            total_pin = 0
            pinRegex = re.compile(r'\s*(\d+)\s+n')
            for subnetinfo in subnets:
                total_pin += int(re.findall(pinRegex, subnetinfo)[0])
                connectedcell = re.findall(adjacentcellRegex, subnetinfo)
                rootcell = connectedcell[0]
                adjacentcell = connectedcell[1:]
                allRootCell.append(rootcell)
                allAdjacentCell.append(adjacentcell)
                allCell.append(connectedcell)
            b = time.time()
            print(f"extract nodes in subnets spends {b - a}s.")
        else:
            print("subnets number != netNumber")
            print("split subnet wrong,extracting nodes in subnets fail")
            exit()

        if total_pin != self._pinNumber:
            print(f"all connected cell number is not {self._pinNumber}, can't save sparse matrix right now.")
            exit()
        return allRootCell, allAdjacentCell, allCell, total_pin

    def construct_sparse_matrix(self, root: list, adjacent: list, all: list) -> csc_matrix:
        """ construct a symmetric sparse matrix,
        where value represents the connection number of corresponding nodes
        :param root: a list of all root nodes
        :param adjacent: a list[list] of adjacent nodes based on root node
        :param all: a list containing all subnet
        :return:
            sparse matrix: csc_matrix
        """
        # initialize sparse matrix with all elements 0
        sparseMatrix = lil_matrix((self._nodesNumber, self._nodesNumber))
        a = time.time()
        # for x, adj in zip(root, adjacent):
        #     for y in adj:
        #         if x == y:
        #             continue  # skip if root = one cell in adjacent cell, make sure the matrix is symmetric
        #         else:
        #             sparseMatrix[self.nodeName2MatrixIndex[x], self.nodeName2MatrixIndex[y]] += 1
        #             sparseMatrix[self.nodeName2MatrixIndex[y], self.nodeName2MatrixIndex[x]] += 1

        # full connection for all cells in the same subnet
        for flag in range(len(all)):  # subnet
            total_cell = len(all[flag])
            for x in all[flag]:
                for y in all[flag]:
                    if x == y:
                        continue  # skip if root = one cell in adjacent cell, make sure the matrix is symmetric
                    else:
                        sparseMatrix[self.nodeName2MatrixIndex[x], self.nodeName2MatrixIndex[y]] += 2 / total_cell
                        # sparseMatrix[self.nodeName2MatrixIndex[y], self.nodeName2MatrixIndex[x]] += 1

        b = time.time()
        print(f"construct sparse matrix spends {b - a}s.")
        sparseMatrix = csc_matrix(sparseMatrix)
        return sparseMatrix

    def net2sparsematrix(self, nodefilepath: Path, netfilepath: Path, sparseMatrixFile: Path,
                         nodeName2MatrixIndex: Path):
        """ represent netlist as sparse matrix
        :param nodefilepath: directory of node info file
        :param netfilepath: directory of net info file
        :param sparseMatrixFile: directory of spares matrix file will be stored
        :param nodeName2MatrixIndex: directory of nodeName2MatrixIndex dictionary will be stored
        :return: None, save sparse matrix
        """
        node_name_info = self.extract_nodes_name(nodefilepath)
        self.construct_nodeName2MatrixIndex_dictionary(node_name_info, nodeName2MatrixIndex)
        subNetList = self.extract_net_info(netfilepath)
        # extract connected nodes from each subnet
        root, adjacent, all, totalpin = self.extract_nodes_in_subnets(subNetList)
        sparse_matrix = self.construct_sparse_matrix(root, adjacent, all)
        save_npz(sparseMatrixFile, sparse_matrix)

    def cluster_level_connectivity(self, clusterResultFile: Path, netlistFile: Path, saveFile: Path):
        """
        :param clusterResultFile:
        :param netlistFile:
        :param saveFile:
        :return:
        """
        # read cluster result file
        clusterResult = read_cluster_result(clusterResultFile)

        # create group-level connectivity matrix
        group_size = max(clusterResult) + 1
        group_level_feature = np.zeros(shape=(group_size, group_size))

        # extract subNets info
        subNetList = self.extract_net_info(netlistFile)
        # extract connected cell in each subnet
        root, adjacent, all, _ = self.extract_nodes_in_subnets(subNetList)
        # get final group-level feature
        # for x, adj in zip(root, adjacent):
        #     for y in adj:
        #         xx = clusterResult[self.nodeName2MatrixIndex[x]]
        #         yy = clusterResult[self.nodeName2MatrixIndex[y]]
        #         if xx != yy:
        #             group_level_feature[xx, yy] += 1
        #             group_level_feature[yy, xx] += 1
        #         else:
        #             pass

        # full connection for all cells in the same subnet
        for flag in range(len(all)):
            for x in all[flag]:
                for y in all[flag]:
                    xx = clusterResult[self.nodeName2MatrixIndex[x]]
                    yy = clusterResult[self.nodeName2MatrixIndex[y]]
                    if xx != yy:
                        group_level_feature[xx, yy] += 1
                        group_level_feature[yy, xx] += 1
                    else:
                        pass

        # save group-level feature matrix
        np.savetxt(saveFile, group_level_feature, fmt='%d', delimiter=',')

    def hmetis_level_connectivity(self, clusterResult, netlistFile: Path, nodefilepath: Path):
        # create group-level connectivity matrix
        group_size = max(clusterResult) + 1
        # group_level_feature = np.zeros(shape=(group_size, group_size))
        group_level_feature = lil_matrix((group_size, group_size))

        # extract subNets info
        subNetList = self.extract_net_info(netlistFile)
        # extract connected cell in each subnet
        root, adjacent, all, _ = self.extract_nodes_in_subnets(subNetList)
        #
        # node_info = self.extract_nodes_name(nodefilepath)
        #
        # for i in range(len(node_info)):
        #     self.nodeName2MatrixIndex.setdefault(node_info[i], i)

        # get final group-level feature
        # for x, adj in zip(root, adjacent):
        #     for y in adj:
        #         xx = clusterResult[self.cellName2MatrixIndex[x]]
        #         yy = clusterResult[self.cellName2MatrixIndex[y]]
        #         if xx != yy:
        #             group_level_feature[xx, yy] += 1
        #             group_level_feature[yy, xx] += 1
        #         else:
        #             pass

        # full connection for all cells in the same subnet
        for flag in range(len(all)):
            for x in all[flag]:
                for y in all[flag]:
                    xx = clusterResult[int(x[1:])]
                    yy = clusterResult[int(y[1:])]
                    if xx != yy:
                        group_level_feature[xx, yy] += 1
                        group_level_feature[yy, xx] += 1
                    else:
                        pass

        # save group-level feature matrix
        # np.savetxt(saveFile, group_level_feature, fmt='%d', delimiter=',')
        group_level_feature = coo_matrix(group_level_feature)
        return group_level_feature

    def cluster_connectivity_macro(self, clusterResultFile: Path, netlistFile: Path, pl_file: Path):
        # 把macro看做单独的类，计算类与类之间的连接数，作为feature输入到GAT中
        # cluster = read_cluster_result(clusterResultFile)
        cluster = np.loadtxt(clusterResultFile, delimiter=',')
        cluster = cluster.astype(int)

        myText = open(pl_file, mode='r')
        fix_node_id = []
        for line in myText.readlines():
            if line.find('FIXED') != -1:
                fix_node_id.append(line.split()[0])
        groupsNumbers = max(cluster) + 1
        fix_node_num = len(fix_node_id)
        cluster_num = groupsNumbers + fix_node_num
        print("cluster_num: ", cluster_num)
        group_level_feature = np.zeros(shape=(cluster_num, cluster_num))

        # take fix nodes as new clusters
        flag = 0
        for node in fix_node_id:
            cluster[int(node[1:])] = groupsNumbers + flag
            flag += 1

        # extract subNets info
        subNetList = self.extract_net_info(netlistFile)
        # extract connected cell in each subnet
        root, adjacent, all, _ = self.extract_nodes_in_subnets(subNetList)

        # full connection for all cells in the same subnet
        # for flag in range(len(all)):
        #     for x in all[flag]:
        #         for y in all[flag]:
        #             xx = cluster[int(x[1:])]
        #             yy = cluster[int(y[1:])]
        #             if xx != yy:
        #                 group_level_feature[xx, yy] += 1
        #                 group_level_feature[yy, xx] += 1
        #             else:
        #                 pass

        for x, adj in zip(root, adjacent):
            for y in adj:
                xx = cluster[int(x[1:])]
                yy = cluster[int(y[1:])]
                if xx != yy:
                    group_level_feature[xx, yy] += 1
                    group_level_feature[yy, xx] += 1
                else:
                    pass

        return group_level_feature, cluster, cluster_num

    def cluster_connectivity_macro_noIO(self, clusterResultFile: Path, netlistFile: Path, pl_file: Path,
                                        node_file: Path):
        # 把macro看做单独的类，IO不看做单独的类，计算类与类之间的连接数，作为feature输入模型中

        cluster = np.loadtxt(clusterResultFile, delimiter=',')
        cluster = cluster.astype(int)

        myText = open(pl_file, mode='r')
        nodeText = open(node_file, mode='r')
        fix_node_id = []

        for line in myText.readlines():
            if line.find('FIXED') != -1:
                fix_node_id.append(line.split()[0])
                # fix_node_pos.append(line.split()[1:3])
        for line in nodeText.readlines():
            if line.find('terminal') != -1:
                fix_node_id.remove(line.split()[0])

        groupsNumbers = max(cluster) + 1
        fix_node_num = len(fix_node_id)
        cluster_num = groupsNumbers + fix_node_num
        print("cluster_num: ", cluster_num)
        group_level_feature = np.zeros(shape=(cluster_num, cluster_num))

        # take fix nodes as new clusters
        flag = 0
        for node in fix_node_id:
            cluster[int(node[1:])] = groupsNumbers + flag
            flag += 1

        # extract subNets info
        subNetList = self.extract_net_info(netlistFile)
        # extract connected cell in each subnet
        root, adjacent, all, _ = self.extract_nodes_in_subnets(subNetList)

        for x, adj in zip(root, adjacent):
            for y in adj:
                xx = cluster[int(x[1:])]
                yy = cluster[int(y[1:])]
                if xx != yy:
                    group_level_feature[xx, yy] += 1
                    group_level_feature[yy, xx] += 1
                else:
                    pass

        return group_level_feature, cluster, cluster_num

    def norm_feature(self, feature, adj):
        feature = (feature - feature.min()) / (feature.max() - feature.min())
        nfeature = feature * adj
        return nfeature


def parser_bookshelf_main(benchmark):
    pa = Parser()
    sourcePath = Path('../data/ispd2005/')
    real_benchmark_nets_path = Path(sourcePath, benchmark, benchmark + '.nets')
    real_benchmark_nodes_path = Path(sourcePath, benchmark, benchmark + '.nodes')
    save_path = '../data/ispd2005_parser/'
    make_dir(save_path)
    sparseMatrixFile = Path(save_path, benchmark + '.npz')
    nodeName2MatrixIndex = Path(save_path, benchmark + '.json')
    pa.net2sparsematrix(real_benchmark_nodes_path, real_benchmark_nets_path, sparseMatrixFile, nodeName2MatrixIndex)


def read_dp_result_pl(benchmark):
    pl_file = 'data/dreamplace_result/' + benchmark + '/' + benchmark + '.gp.pl'
    myText = open(pl_file, mode='r')
    location = []
    fixed_cell = []
    flag = 0
    for line in myText.readlines():
        if flag > 1:
            if line.find('FIXED') != -1:
                fixed_cell.append([int(line.split()[1]), int(line.split()[2])])
            else:
                location.append([int(line.split()[1]), int(line.split()[2])])
        flag += 1
    return location, fixed_cell


def read_original_pl_file(benchmark):
    # get the number of movable cell and fixed cell
    # get the location of fixed cell
    pl_file = 'data/ispd2005/' + benchmark + '/' + benchmark + '.pl'
    myText = open(pl_file, mode='r')
    fixed_cell_location = []
    movable_num = 0
    fixed_num = 0
    flag = 0
    for line in myText.readlines():
        if flag > 3:
            if line.find('FIXED') != -1:
                fixed_cell_location.append([int(line.split()[1]), int(line.split()[2])])
                fixed_num += 1
            else:
                movable_num += 1
        flag += 1
    fixed_cell_location = np.array(fixed_cell_location)
    return fixed_cell_location, movable_num, fixed_num


def find_fixed_point_def(file):
    io_id = []
    io_pos = []
    with open(file, 'r') as f:
        info = f.read()
        totalCellNumber = int(re.search(r'COMPONENTS\s(\d+)\s;', info).group(1))

        # read PIN(IO Pad) info
        PINSRegex = re.compile(r'pins\s+(\d+)', re.IGNORECASE)
        totalPinNumber = int(re.search(PINSRegex, info).group(1)) - 1  # 去掉clk pin

        PINInfo = info[info.find('PINS'):info.find('END PINS')]
        PINList = re.split(r';', PINInfo)
        PINList.pop(0)
        PINList.pop(-1)

        for i in range(totalPinNumber):
            io_id.append(i + totalCellNumber)
            pos_info = PINList[i].split('\n')[3]
            io_pos.append([int(pos_info.split()[3]), int(pos_info.split()[4])])
    io_pos = np.array(io_pos)

    return totalCellNumber, totalPinNumber, io_id, io_pos


def placement_region(fixed_pos):
    xf = fixed_pos[:, 0]
    yf = fixed_pos[:, 1]
    x_min = np.min(xf)
    x_max = np.max(xf)
    y_min = np.min(yf)
    y_max = np.max(yf)
    print('placement region: ', x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max


def generate_initial_locations(fixed_cell_location, movable_num, scale):
    x_min, y_min, x_max, y_max = placement_region(fixed_cell_location)
    random_initial = np.random.rand(int(movable_num), 2)
    xcenter = (x_max - x_min) / 2 + x_min
    ycenter = (y_max - y_min) / 2 + y_min
    random_initial[:, 0] = ((random_initial[:, 0] - 0.5) * (x_max - x_min) * scale) + xcenter
    random_initial[:, 1] = ((random_initial[:, 1] - 0.5) * (y_max - y_min) * scale) + ycenter
    return random_initial

def generate_initial_locations_without_fixed(movable_num, scale):
    x_min = 0
    y_min = 0
    x_max = 10000
    y_max = 10000
    random_initial = np.random.rand(int(movable_num), 2)
    xcenter = (x_max - x_min) / 2 + x_min
    ycenter = (y_max - y_min) / 2 + y_min
    random_initial[:, 0] = ((random_initial[:, 0] - 0.5) * (x_max - x_min) * scale) + xcenter
    random_initial[:, 1] = ((random_initial[:, 1] - 0.5) * (y_max - y_min) * scale) + ycenter
    return random_initial


def calLaplacianMatrix(adjacentMatrix):
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    norm = np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
    return norm


def spectral_decomposition(adj):
    laplacianmatrix = calLaplacianMatrix(adj)
    eigenvalue, eigenvector = linalg.eigsh(laplacianmatrix, which='SM')
    # eigenvalue, eigenvector = linalg.eigsh(laplacianmatrix)

    pos = eigenvector[:, 1:]
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]

    return pos, pos_x, pos_y, eigenvalue



