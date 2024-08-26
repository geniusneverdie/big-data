import numpy as np
import copy
import os
import json
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import gmres

def prepare_block_data(num_blocks=10):
    data_path = 'Data.txt'
    node_degrees = {}
    with open(data_path, 'r') as file:
        for line in file:
            source, target = map(int, line.strip().split())
            if source not in node_degrees.keys():
                node_degrees[source] = 0
            if target not in node_degrees.keys():
                node_degrees[target] = 0
            node_degrees[source] += 1
    total_nodes = len(node_degrees)
    sorted_node_degrees = list(sorted(node_degrees.items(), key=lambda x: x[0]))
    node_indices = {node[0]: i for i, node in enumerate(sorted_node_degrees)}

    print("-------- Start to load blocks ---------")

    nodes_per_block = total_nodes // num_blocks
    empty_dict = {node: [] for node in node_degrees}
    block_data = [copy.deepcopy(empty_dict) for _ in range(num_blocks)]

    with open(data_path, 'r') as f:
        for line in f:
            source, target = map(int, line.strip().split())
            block_index = (node_indices[target] // nodes_per_block) if (node_indices[target] // nodes_per_block) < (num_blocks) else num_blocks - 1
            block_data[block_index][source].append(target)

    if not os.path.exists("Block_matrix"):
        os.mkdir("Block_matrix")

    print('--------- Saving matrices ----------')
    for block in range(num_blocks):
        save_matrix(block_data[block], block)

    print("-------- All blocks loaded! ---------")
    return sorted_node_degrees, node_indices

def save_matrix(data, index):
    matrix_filename = 'Block_matrix\\Block' + str(index) + '.matrix'
    with open(matrix_filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f)

def load_matrix(index):
    matrix_filename = 'Block_matrix\\Block' + str(index) + '.matrix'
    with open(matrix_filename, 'r+', encoding='utf-8') as f:
        return json.load(f)

def save_vector(data, index, is_new=False):
    suffix = '.new' if is_new else '.old'
    vector_filename = 'RVector\\r' + str(index) + suffix
    with open(vector_filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f)

def load_vector(index, is_new=False):
    suffix = '.new' if is_new else '.old'
    vector_filename = 'RVector\\r' + str(index) + suffix
    with open(vector_filename, 'r+', encoding='utf-8') as f:
        return json.load(f)

def load_data(file_path):
    adjacency_list = {}
    node_set = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            source_node, target_node = map(int, line.strip().split())
            node_set.add(source_node)
            node_set.add(target_node)
            if source_node not in adjacency_list.keys():
                adjacency_list[source_node] = []
            adjacency_list[source_node].append(target_node)

    return adjacency_list, node_set

def initialize_r(node_degrees, num_blocks=10, damping_factor=0.85):
    print("----------- Initializing vector -----------")
    if not os.path.exists("RVector"):
        os.mkdir("RVector")
    total_nodes = len(node_degrees)
    nodes_per_block = total_nodes // num_blocks
    r_old = np.ones(total_nodes) / total_nodes
    r_new = np.array([(1 - damping_factor) / total_nodes for _ in range(total_nodes)])
    for block in range(num_blocks):
        if block < num_blocks - 1:
            save_vector(r_old[block * nodes_per_block: block * nodes_per_block + nodes_per_block].tolist(), block, False)
            save_vector(r_new[block * nodes_per_block: block * nodes_per_block + nodes_per_block].tolist(), block, True)
        else:
            save_vector(r_old[block * nodes_per_block:].tolist(), block, False)
            save_vector(r_new[block * nodes_per_block:].tolist(), block, True)
    print("----------- Vector is ready！------------")

def calculate_block_rank(node_degrees, node_indices, num_blocks=10, damping_factor=0.85, tol=1e-8, iter_num=0):
    total_nodes = len(node_degrees)
    while True:
        theta = 0
        iter_num += 1

        for block in range(num_blocks):
            graph = load_matrix(block)
            r_new = np.array(load_vector(block, True))
            block_base = total_nodes // num_blocks * block

            for p_block in range(num_blocks):
                base = total_nodes // num_blocks * p_block
                r_old = np.array(load_vector(p_block))

                for rx, weight_rx in enumerate(r_old):
                    rx_ind = rx + base
                    rx_name = str(node_degrees[rx_ind][0])
                    degree_rx = node_degrees[rx_ind][1]
                    if degree_rx > 0:
                        for destination in graph[rx_name]:
                            des_idx = node_indices[destination] - block_base
                            r_new[des_idx] = (damping_factor * weight_rx) / degree_rx + r_new[des_idx]

                    else:
                        r_new += damping_factor / total_nodes * weight_rx

            r_old = np.array(load_vector(block))
            save_vector(r_new.tolist(), block, False)

            theta += np.abs(r_new - r_old).sum()
        if theta < tol:
            print(f'Iterated {iter_num} times')
            break

def calculate_page_rank(adjacency_list, all_nodes, damping_factor=0.85, tol=1e-8, iter_num=0):
    total_nodes = len(all_nodes)
    all_nodes = list(sorted(all_nodes))
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    r_old = np.ones(total_nodes) / total_nodes
    while True:
        iter_num += 1
        r_new = np.array([(1 - damping_factor) / total_nodes for _ in range(total_nodes)])
        for rx_ind, weight_rx in enumerate(r_old):
            rx_name = all_nodes[rx_ind]
            if rx_name in adjacency_list.keys():
                degree_rx = len(adjacency_list[rx_name])
                for destination in adjacency_list[rx_name]:
                    des_idx = node_indices[destination]
                    r_new[des_idx] += damping_factor * weight_rx / degree_rx

            else:
                degree_rx = total_nodes
                r_new += damping_factor * 1 / degree_rx * weight_rx
        S = r_new.sum()
        r_new += (1-S) / total_nodes
        if np.abs(r_new - r_old).sum() < tol * len(r_old):
            print(f'Iterated {iter_num} times')
            break
        r_old = r_new

    return r_old, node_indices

#共轭梯度法
def calculate_page_rank1(adjacency_list, all_nodes, damping_factor=0.85, tol=1e-8, iter_num=0):
    total_nodes = len(all_nodes)
    all_nodes = list(sorted(all_nodes))
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    r_old = np.ones(total_nodes) / total_nodes
    r_new = np.array([(1 - damping_factor) / total_nodes for _ in range(total_nodes)])

    # 构建转移矩阵
    transition_matrix = np.zeros((total_nodes, total_nodes))
    for rx_ind, weight_rx in enumerate(r_old):
        rx_name = all_nodes[rx_ind]
        if rx_name in adjacency_list.keys():
            degree_rx = len(adjacency_list[rx_name])
            for destination in adjacency_list[rx_name]:
                des_idx = node_indices[destination]
                transition_matrix[des_idx, rx_ind] = damping_factor * weight_rx / degree_rx

    # 使用共轭梯度法求解线性系统
    r_new, info = cg(transition_matrix, r_new, tol=tol)
    if info != 0:
        print("Conjugate gradient method did not converge")

    return r_new, node_indices

#GMRES方法
def calculate_page_rank2(adjacency_list, all_nodes, damping_factor=0.85, tol=1e-8, iter_num=0):
    total_nodes = len(all_nodes)
    all_nodes = list(sorted(all_nodes))
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    r_old = np.ones(total_nodes) / total_nodes
    r_new = np.array([(1 - damping_factor) / total_nodes for _ in range(total_nodes)])

    # 构建转移矩阵
    transition_matrix = np.zeros((total_nodes, total_nodes))
    for rx_ind, weight_rx in enumerate(r_old):
        rx_name = all_nodes[rx_ind]
        if rx_name in adjacency_list.keys():
            degree_rx = len(adjacency_list[rx_name])
            for destination in adjacency_list[rx_name]:
                des_idx = node_indices[destination]
                transition_matrix[des_idx, rx_ind] = damping_factor * weight_rx / degree_rx

    # 使用 GMRES 方法求解线性系统
    r_new, info = gmres(transition_matrix, r_new, tol=tol)
    if info != 0:
        print("GMRES method did not converge")

    return r_new, node_indices

def get_top_nodes_block(node_indices, num_blocks=10, top_node_count=100):
    total_nodes = len(node_indices)
    top = [0] * total_nodes
    for block in range(num_blocks):
        base = total_nodes // num_blocks * block
        if block == num_blocks - 1:
            top[base:] = load_vector(block)
        else:
            top[base:base+total_nodes//num_blocks] = load_vector(block)

    sorted_nodes = sorted(node_indices.items(), key=lambda x: top[x[1]], reverse=True)
    return [(node, top[index]) for node, index in sorted_nodes[:top_node_count]]

def get_top_nodes(page_rank_vector, node_indices, top_node_count=100):
    sorted_nodes = sorted(node_indices.items(), key=lambda x: page_rank_vector[x[1]], reverse=True)
    return [(node, page_rank_vector[index]) for node, index in sorted_nodes[:top_node_count]]

def save_top_100_nodes():
    if not os.path.exists("results"):
        os.mkdir("results")
    with open("results\\block_result.txt", 'w+') as f:
        for i, x in enumerate(top_100_nodes):
            f.write(f'{x[0]} {x[1]}\n')
    print('Save finished')

def save_top_100_nodes1():
    if not os.path.exists("results"):
        os.mkdir("results")
    with open("results\\block_result1.txt", 'w+') as f:
        for i, x in enumerate(top_100_nodes):
            f.write(f'{x[0]} {x[1]}\n')
    print('Save finished')

def save_top_100_nodes2():
    if not os.path.exists("results"):
        os.mkdir("results")
    with open("results\\block_result2.txt", 'w+') as f:
        for i, x in enumerate(top_100_nodes):
            f.write(f'{x[0]} {x[1]}\n')
    print('Save finished')

if __name__ == '__main__':
    damping_factor = 0.85
    node_degrees, node_indices = prepare_block_data()
    initialize_r(node_degrees)
    calculate_block_rank(node_degrees, node_indices)
    top_100_nodes = get_top_nodes_block(node_indices)
    print(f'Top 100 Nodes with their PageRank scores (damping factor = {damping_factor}):')
    for node, rank in top_100_nodes:
        print(f'NodeID: {node}, PageRank: {rank}')
    save_top_100_nodes()
    damping_factor = 0.85
    node_degrees, node_indices = prepare_block_data()
    initialize_r(node_degrees)
    calculate_block_rank(node_degrees, node_indices)
    top_100_nodes = get_top_nodes_block(node_indices)
    print(f'Top 100 Nodes with their PageRank scores (damping factor = {damping_factor}):')
    for node, rank in top_100_nodes:
        print(f'NodeID: {node}, PageRank: {rank}')
    save_top_100_nodes1()
    damping_factor = 0.85
    node_degrees, node_indices = prepare_block_data()
    initialize_r(node_degrees)
    calculate_block_rank(node_degrees, node_indices)
    top_100_nodes = get_top_nodes_block(node_indices)
    print(f'Top 100 Nodes with their PageRank scores (damping factor = {damping_factor}):')
    for node, rank in top_100_nodes:
        print(f'NodeID: {node}, PageRank: {rank}')
    save_top_100_nodes2()