import os
import time
import sys
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg, identity
import utils.util as util

sys.path.append(os.getcwd())
from utils.mix_frequency_filter import GF_CF
import matplotlib.pyplot as plt
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_pos(pos):
    plt.scatter(pos[:, 0], pos[:, 1])
    # plt.ylabel(u'x', fontsize=15)
    # plt.xlabel(u'y', fontsize=15)
    # plt.xlim((0,100))
    # fig = plt.gcf()
    # fig.savefig('7result_analysis/'+str(node)+'.png',format='png',pad_inches = 0)
    plt.show()


def plot_pos2(pos, fixp):
    # fixed node are placed at predetermined locations
    plt.scatter(pos[:, 0], pos[:, 1], color='lightsteelblue', label='Movable cell')
    plt.scatter(fixp[:, 0], fixp[:, 1], color='indianred', label='Fixed cell', marker='s')
    # plt.xlim(0, 11092)
    # plt.ylim(0, 11092)
    plt.legend(prop={'size': 15})
    plt.tick_params(labelsize=13)
    fig = plt.gcf()
    fig.savefig('paper_plot/plot_iteration_fig3/gift_location.pdf', format='pdf', pad_inches=0)
    fig.savefig('paper_plot/plot_iteration_fig3/gift_location.png', format='png', pad_inches=0)
    plt.show()


def plot_pos3(pos, fixp):
    # fixed node are placed at the predicted locations
    plt.scatter(pos[:, 0], pos[:, 1], color='b')
    plt.scatter(pos[-len(fixp):, 0], pos[-len(fixp):, 1], color='r')
    plt.show()


def plot_color(x, y):
    color_list = ['red', 'yellow', 'blue', 'black', 'grey', 'c', 'green', 'm']
    plt.scatter(x, y, marker='o', s=50, color='darkblue')
    plt.ylabel(u'x', fontsize=15)
    plt.xlabel(u'y', fontsize=15)
    for i in range(len(x)):
        if i < 8:
            plt.scatter(x[i], y[i], color=color_list[i])
        else:
            continue

    # plt.scatter(x[:2], y[:2], color='blue', marker='o')
    plt.show()


def write_pl_file(movable_pos, fixed_pos, file_path):
    myText = open(file_path + 'graph_signal.pl', 'w')
    myText.write("UCLA pl 1.0\n" + "# Created	:	Jan  6 2005\n" +
                 "# User   	:	Gi-Joon Nam & Mehmet Yildiz at IBM Austin Research({gnam, mcan}@us.ibm.com)\n\n")
    for i in range(len(movable_pos)):
        data = "o" + str(i) + "  "
        xcoordinate = str(movable_pos[i][0])
        ycoordinate = str(movable_pos[i][0])
        data = data + xcoordinate + "   " + ycoordinate + "    :    N"
        myText.write(data + '\n')
    for i in range(len(fixed_pos)):
        data = "o" + str(i) + "  "
        xcoordinate = str(fixed_pos[i][0])
        ycoordinate = str(fixed_pos[i][0])
        data = data + xcoordinate + "   " + ycoordinate + "    :    N /FIXED"
        myText.write(data + '\n')
    myText.close()


def write_positions_to_def(defFile, position, saveFile):
    read_file = open(defFile, 'r')
    save_file = open(saveFile + 'graph_signal_lmh.def', 'w')
    flag = 0
    for line in read_file.readlines():
        if line.find('UNPLACED') != -1:
            line = line.replace('UNPLACED',
                                'PLACED ( ' + str(int(position[flag][0])) + ' ' + str(int(position[flag][1])) + ' ) N',
                                1)

            save_file.write(line)
            flag += 1
        else:
            save_file.write(line)
    save_file.close()


def bookshelf_main(benchmark_list):
    for benchmark_name in benchmark_list:
        # 以下两步就是把netlist构建成图，存成稀疏矩阵的形式，放到adj里
        adj_npz = load_npz('data/ispd2005_parser/' + benchmark_name + '.npz')
        adj = csc_matrix(adj_npz)
        print('adj shape: ', adj.shape)

        # -----------------读取固定点位置和移动点个数 ------------------------#
        # 如果没有固定点，可以都当做移动点
        fixed_cell_location, movable_num, fixed_num = util.read_original_pl_file(benchmark_name)
        print('movable cell num: ', movable_num, 'fixed cell num: ', fixed_num)

        # --------------------生成移动点的初始位置 ------------------------#
        # 初始位置就是随机放到布局区域中心，随机即可，但每个cell的横纵坐标不能一样
        scale = 0.6  # 初始位置的分布范围
        random_initial = util.generate_initial_locations(fixed_cell_location, movable_num, scale)
        random_initial = np.concatenate((random_initial, fixed_cell_location), 0)
        print(random_initial.shape)
        # plot_pos3(random_initial, fixed_cell_location)

        # ----------- low-pass filter ------------------------#
        low_pass_filter = GF_CF(adj)
        low_pass_filter.train(4)
        location_low = low_pass_filter.get_cell_position(2, random_initial)
        print('finish low-pass filter')

        # ----------- m-pass filter ------------------------#
        # low_pass_filter.train(2)
        # location_m = low_pass_filter.get_cell_position(2, random_initial)
        # print('finish m-pass filter')

        # ----------- h-pass filter ------------------------#
        # low_pass_filter.train(1)
        # location_h = low_pass_filter.get_cell_position(1, random_initial)
        # print('finish h-pass filter')

        # ------------ final location  ------------------------#
        # location = 0.8 * location_low + 0.1 * location_m + 0.1 * location_h
        location = location_low
        # 可以先只用low-pass filter

        # ---------------- plot ------------------------#
        plot_pos3(location_low, fixed_cell_location)
        # plot_pos3(location_m, fixed_cell_location)
        # plot_pos3(location_h, fixed_cell_location)
        # plot_pos3(location, fixed_cell_location)

        # ------------------write pl file ------------------------#
        save_path = 'data/ispd2005_parser/filter_result2/' + benchmark_name + '/'
        util.make_dir(save_path)
        write_pl_file(location[:movable_num, :], fixed_cell_location, save_path)
        # plot_pos2(location, fixed_cell_location)


# def deflef_main(benchmark_list):
#     for benchmark_name in benchmark_list:
#         print('-----------', benchmark_name, '----------------')
#         # 以下两步就是把netlist构建成图，存成稀疏矩阵的形式，放到adj里
#         adj_npz = load_npz('data/ispd2014_parser/' + benchmark_name + '/forQua_conn.npz')
#         adj = csc_matrix(adj_npz)
#         print(adj.size)

#         # -----------------读取固定点位置和移动点个数 ------------------------#
#         # 如果没有固定点，可以都当做移动点
#         # known position 
#         ori_def_file = 'data/ispd2014/' + benchmark_name + '/floorplan.def'
#         def_file = 'data/ispd2014/' + benchmark_name + '/mfloorplan.def'
#         movable_num, fixed_num, fixed_point_ID, fixed_cell_location = util.find_fixed_point_def(def_file)  # 找到固定点及其位置
#         print('movable cell num: ', movable_num, 'fixed cell num: ', fixed_num)

#         # --------------------生成移动点的初始位置 ------------------------#
#         # 初始位置就是随机放到布局区域中心，随机即可，但每个cell的横纵坐标不能一样
#         scale = 0.6  # 初始位置的分布范围
#         # random_initial = util.generate_initial_locations_without_fixed(movable_num, scale)
#         random_initial = util.generate_initial_locations(fixed_cell_location, movable_num, scale)
#         print(random_initial.size)
#         random_initial = np.concatenate((random_initial, fixed_cell_location), 0)
#         print(random_initial.size)
#         plot_pos3(random_initial, fixed_cell_location)

#         # ----------- low-pass filter ------------------------#
#         start = time.time()
#         low_pass_filter = GF_CF(adj)
#         low_pass_filter.train(4)  # 这个值可调
#         location_low = low_pass_filter.get_cell_position(4, random_initial)  # 这个值可调
#         # print('finish low-pass filter!')

#         # ----------- m-pass filter ------------------------#
#         low_pass_filter.train(2)
#         location_m = low_pass_filter.get_cell_position(4, random_initial)
#         # print('finish m-pass filter!')

#         # ----------- h-pass filter ------------------------#
#         low_pass_filter.train(2)
#         location_h = low_pass_filter.get_cell_position(4, random_initial)
#         # print('finish h-pass filter!')
#         end = time.time()
#         print('-------------total time--------', end - start)

#         # ------------ final location  ------------------------#
#         location = 0.8 * location_low + 0.1 * location_m + 0.1 * location_h
#         # location = location_low
#         # 可以先只用low-pass filter

#         # ---------------- plot ------------------------#
#         # plot_pos2(location_low, fixed_cell_location)
#         # plot_pos2(location_m, fixed_cell_location)
#         # plot_pos2(location_h, fixed_cell_location)
#         plot_pos2(location, fixed_cell_location)

#         # ------------------write def file ------------------------#
#         save_path = 'data/data/' + benchmark_name + '/'
#         util.make_dir(save_path)
#         write_positions_to_def(ori_def_file, location[:movable_num, :],
#                                save_path)  # customized save name
        
def deflef_main(benchmark_name):
    print('-----------', benchmark_name, '----------------')
    # 以下两步就是把netlist构建成图，存成稀疏矩阵的形式，放到adj里
    adj_npz = load_npz('./data/benchmark_parser/' + benchmark_name + '/forQua_conn.npz')
    adj = csc_matrix(adj_npz)

    # -----------------读取固定点位置和移动点个数 ------------------------#
    # 如果没有固定点，可以都当做移动点
    # known position 
    ori_def_file = 'data/benchmark/' + benchmark_name + '/floorplan.def'
    def_file = 'data/benchmark/' + benchmark_name + '/mfloorplan.def'
    movable_num, fixed_num, fixed_point_ID, fixed_cell_location = util.find_fixed_point_def(def_file)  # 找到固定点及其位置
    print('movable cell num: ', movable_num, 'fixed cell num: ', fixed_num)

    # --------------------生成移动点的初始位置 ------------------------#
    # 初始位置就是随机放到布局区域中心，随机即可，但每个cell的横纵坐标不能一样
    scale = 0.6  # 初始位置的分布范围
    # random_initial = util.generate_initial_locations_without_fixed(movable_num, scale)
    random_initial = util.generate_initial_locations(fixed_cell_location, movable_num, scale)
    random_initial = np.concatenate((random_initial, fixed_cell_location), 0)
    # plot_pos3(random_initial, fixed_cell_location)

    # ----------- low-pass filter ------------------------#
    start = time.time()
    low_pass_filter = GF_CF(adj)
    low_pass_filter.train(4)  # 这个值可调
    location_low = low_pass_filter.get_cell_position(4, random_initial)  # 这个值可调
    # print('finish low-pass filter!')

    # ----------- m-pass filter ------------------------#
    low_pass_filter.train(2)
    location_m = low_pass_filter.get_cell_position(4, random_initial)
    # print('finish m-pass filter!')

    # ----------- h-pass filter ------------------------#
    low_pass_filter.train(2)
    location_h = low_pass_filter.get_cell_position(4, random_initial)
    # print('finish h-pass filter!')
    end = time.time()
    print('-------------total time--------', end - start)

    # ------------ final location  ------------------------#
    location = 0.8 * location_low + 0.1 * location_m + 0.1 * location_h
    # location = location_low
    # 可以先只用low-pass filter

    # ---------------- plot ------------------------#
    # plot_pos2(location_low, fixed_cell_location)
    # plot_pos2(location_m, fixed_cell_location)
    # plot_pos2(location_h, fixed_cell_location)
    # plot_pos2(location, fixed_cell_location)

    # ------------------write def file ------------------------#
    save_path = 'data/data/' + benchmark_name + '/'
    util.make_dir(save_path)
    write_positions_to_def(ori_def_file, location[:movable_num, :],
                               save_path)  # customized save name


if __name__ == '__main__':
    # benchmark_name_list_ispd2005 = ['bigblue3','bigblue2']
    # benchmark_name_list_ispd2014 = ['mgc_des_perf_1', 'mgc_des_perf_2', 'mgc_edit_dist_1', 'mgc_edit_dist_2',
    #                                 'mgc_matrix_mult', 'mgc_fft', 'mgc_pci_bridge32_1', 'mgc_pci_bridge32_2']
    # benchmark_name_list_epfl = ['adder','arbiter','bar','cavlc','ctrl','dec','div','hyp','i2c','int2float','log2','max','mem_ctrl','multiplier','priority','router','sin','sqrt','square','voter']
    # benchmark_name_list_epfl = ['C17']

    # bookshelf_main(benchmark_name_list_ispd2005) # bookshelf格式
    benchmark_name = sys.argv[1]
    deflef_main(benchmark_name)  # deflef格式
