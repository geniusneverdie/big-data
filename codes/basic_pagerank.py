import numpy as np
import os
from scipy.sparse.linalg import LinearOperator, gmres

data_file_path = 'Data.txt'
damping_factor = 0.85
convergence_threshold = 1e-12
max_iterations = 1000  # 增加迭代次数

def load_graph_data(file_path):
    """
    从文件中读取数据，构建图并返回图数据和所有节点的集合。
    """
    adjacency_list = {}
    node_set = set()

    with open(file_path, 'r') as file:
        for line in file:
            source_node, target_node = map(int, line.strip().split())
            node_set.add(source_node)
            node_set.add(target_node)
            if source_node not in adjacency_list:
                adjacency_list[source_node] = []
            adjacency_list[source_node].append(target_node)

    return adjacency_list, node_set

def compute_page_rank(adjacency_list, node_set, damping_factor, convergence_threshold, max_iterations):
    """
    基于给定的图、节点和参数，计算PageRank值。
    """
    node_count = len(node_set)
    node_index = {node: i for i, node in enumerate(sorted(node_set))}

    # 构建转移矩阵S
    transition_matrix = np.zeros([node_count, node_count], dtype = np.float64)  # 使用更精确的数值类型
    for source_node, target_nodes in adjacency_list.items():
        for target_node in target_nodes:
            transition_matrix[node_index[target_node], node_index[source_node]] = 1

    # 处理矩阵
    for col in range(node_count):
        if (col_sum := transition_matrix[:, col].sum()) == 0:
            transition_matrix[:, col] = 1 / node_count
        else:
            transition_matrix[:, col] /= col_sum

    # 计算总转移矩阵M
    E = np.ones((node_count, node_count), dtype = np.float64)
    M = damping_factor * transition_matrix + (1 - damping_factor) / node_count * E

    # 迭代更新PageRank值，直到收敛
    page_rank_vector = np.ones(node_count, dtype = np.float64) / node_count
    diff, iteration = float('inf'), 1
    while diff > convergence_threshold and iteration <= max_iterations:
        new_page_rank_vector = M @ page_rank_vector
        diff = np.linalg.norm(new_page_rank_vector - page_rank_vector)
        print('Iteration: {}, diff: {}'.format(iteration, diff))
        page_rank_vector = new_page_rank_vector + (1 - new_page_rank_vector.sum()) / node_count
        iteration += 1

    return page_rank_vector, node_index


def compute_page_rank2(adjacency_list, node_set, damping_factor, convergence_threshold, max_iterations):
    """
    基于给定的图、节点和参数，计算PageRank值。
    """
    node_count = len(node_set)
    node_index = {node: i for i, node in enumerate(sorted(node_set))}

    # 构建转移矩阵S
    transition_matrix = np.zeros([node_count, node_count], dtype = np.float64)  # 使用更精确的数值类型
    for source_node, target_nodes in adjacency_list.items():
        for target_node in target_nodes:
            transition_matrix[node_index[target_node], node_index[source_node]] = 1

    # 处理矩阵
    for col in range(node_count):
        if (col_sum := transition_matrix[:, col].sum()) == 0:
            transition_matrix[:, col] = 1 / node_count
        else:
            transition_matrix[:, col] /= col_sum

    # 计算总转移矩阵M
    E = np.ones((node_count, node_count), dtype = np.float64)
    M = damping_factor * transition_matrix + (1 - damping_factor) / node_count * E

    # 创建对角线预条件器
    preconditioner = np.diag(1 / np.diag(M))

    # 创建线性运算符
    linear_operator = LinearOperator(shape=M.shape, matvec=lambda x: M @ x)

    # 迭代更新PageRank值，直到收敛
    page_rank_vector = np.ones(node_count, dtype = np.float64) / node_count
    diff, iteration = float('inf'), 1
    while diff > convergence_threshold and iteration <= max_iterations:
        new_page_rank_vector, info = gmres(linear_operator, page_rank_vector, M=preconditioner,
                                           tol=convergence_threshold)
        if info != 0:
            print("GMRES method did not converge")
            break
        diff = np.linalg.norm(new_page_rank_vector - page_rank_vector)
        print('Iteration: {}, diff: {}'.format(iteration, diff))
        page_rank_vector = new_page_rank_vector + (1 - new_page_rank_vector.sum()) / node_count
        iteration += 1

    return page_rank_vector, node_index
def get_top_ranked_nodes(page_rank_vector, node_index, top_node_count = 100):
    """
    返回前100个具有最高PageRank值的节点及其PageRank值。
    """
    sorted_nodes = sorted(node_index.items(), key = lambda x: page_rank_vector[x[1]], reverse = True)
    return [(node, page_rank_vector[index]) for node, index in sorted_nodes[:top_node_count]]

def save_results(file_path, top_ranked_nodes):
    """
    将结果写入文件。
    """
    if not os.path.exists("results"):
        os.mkdir("results")
    with open(file_path, 'w') as file:
        for node, rank in top_ranked_nodes:
            file.write(f'{node} {rank}\n')

def save_results2(file_path, top_ranked_nodes):
    """
    将结果写入文件。
    """
    if not os.path.exists("results"):
        os.mkdir("results")
    with open(file_path, 'w') as file:
        for node, rank in top_ranked_nodes:
            file.write(f'{node} {rank}\n')

def main():
    adjacency_list, node_set = load_graph_data(data_file_path)
    page_rank_vector, node_index = compute_page_rank(adjacency_list, node_set, damping_factor, convergence_threshold, max_iterations)
    top_ranked_nodes = get_top_ranked_nodes(page_rank_vector, node_index)
    save_results('results\\basic_result.txt', top_ranked_nodes)
    print(f'Top 100 Nodes with their PageRank scores (damping factor = {damping_factor}):')
    for node, rank in top_ranked_nodes:
        print(f'NodeID: {node}, PageRank: {rank}')
    adjacency_list, node_set = load_graph_data(data_file_path)
    page_rank_vector2, node_index2 = compute_page_rank2(adjacency_list, node_set, damping_factor, convergence_threshold, max_iterations)
    top_ranked_nodes2 = get_top_ranked_nodes(page_rank_vector2, node_index2)
    save_results2('results\\basic_result2.txt', top_ranked_nodes2)
    print(f'Top 100 Nodes with their PageRank scores (damping factor = {damping_factor}):')
    for node, rank in top_ranked_nodes2:
        print(f'NodeID: {node}, PageRank: {rank}')
    return

if __name__ == '__main__':
    main()