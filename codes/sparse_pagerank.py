import os
import numpy as np
from collections import defaultdict


class SparseGraph:
    def __init__(self, dtype=np.float64):
        """
        初始化稀疏图。
        参数:
            data (dict): 一个表示图的字典，形式是{source_node: (outdegree, [destination_nodes])}，默认为 None。
            dtype: 数据类型，默认为 np.float64。
        """
        self.data = defaultdict(lambda: (0, []))
        self.all_nodes = set()
        self.dtype = dtype

    def __getitem__(self, source):
        """
        通过下标访问源节点的出度和目标节点列表。
        参数:
            source: 源节点。

        返回:
            tuple: 一个元组，包含源节点的出度和目标节点列表。
        """
        return self.data[source]

    def __contains__(self, source):
        """
        判断源节点是否在图中。
        参数:
            source: 源节点。

        返回:
            bool: 如果源节点在图中，返回True，否则返回False。
        """
        return source in self.data

    def __str__(self):
        """
        将稀疏图转换为字符串表示。
        返回:
            str: 稀疏图的字符串表示。
        """
        return f"SparseGraph(data={self.data})"

    def add_edge(self, source, destination):
        """
        添加一条从源节点到目标节点的边。
        参数:
            source: 源节点。
            destination: 目标节点。
        """
        self.all_nodes.add(source)
        self.all_nodes.add(destination)
        if destination not in self.data:
            self.data[destination] = (0, [])

        if source not in self.data:
            self.data[source] = (1, [destination])

        else:
            outdegree, destinations = self.data[source]
            destinations.append(destination)
            self.data[source] = (outdegree + 1, destinations)

    def get_outdegree(self, source):
        """
        获取源节点的出度。
        参数
            source: 源节点。
        返回:
            int: 源节点的出度。
        """
        return self.data[source][0] if source in self.data else 0

    def get_destinations(self, source):
        """
        获取源节点连接的目标节点列表。
        参数:
            source: 源节点。
        返回:
            list: 目标节点列表。
        """
        return self.data[source][1] if source in self.data else []

    def get_sources(self):
        """
        获取所有源节点。
        返回:
            list: 源节点列表。
        """
        return list(self.data.keys())

    def get_no_outdegree_nodes(self):
        """
        获取出度为0的节点。
        返回:
            list: 出度为0的节点列表。
        """
        return [node for node in self.data if self.data[node][0] == 0]

    def get_all_nodes(self):
        """
        返回:
            list: 所有节点的列表。
        """
        return list(self.all_nodes)


file_path = 'Data.txt'
teleport_parameter = 0.85
tol = 1e-12

# 从文件中读取数据并构建图
def read_data(file_path):
    G = SparseGraph()
    with open(file_path, 'r') as file:
        for line in file:
            from_node, to_node = map(int, line.strip().split())
            G.add_edge(from_node, to_node)
    return G

# PageRank 算法的稀疏实现
def page_rank_sparse(G, beta, tol):
    N = len(G.all_nodes)
    ranks = np.array([1 / N] * N, dtype = np.float64)
    node_idx_map = {node: i for i, node in enumerate(G.all_nodes)}
    diff, iteration = float('inf'), 1

    while diff > tol:
        new_ranks = np.array([(1 - beta) / N] * N, dtype = np.float64)
        for node, (out_degree, dests) in G.data.items():
            if out_degree == 0:
                new_ranks += (beta * ranks[node_idx_map[node]]) / N
                continue
            rank_contribution = beta * ranks[node_idx_map[node]] / out_degree
            for dest in dests:
                new_ranks[node_idx_map[dest]] += rank_contribution
                
        new_ranks /= new_ranks.sum()
        diff = np.sum(np.abs(ranks - new_ranks))
        print(f'Iteration {iteration}: diff = {diff}')
        ranks = new_ranks
        iteration += 1

    return ranks

# 返回具有最高 PageRank 分数的节点
def top_nodes(pr, all_nodes, num_top_nodes=100):
    node_pr = list(zip(all_nodes, pr))
    sorted_nodes = sorted(node_pr, key=lambda x: x[1], reverse=True)
    return sorted_nodes[:num_top_nodes]

# 将结果写入文件
def write_result(file_path, top_100_nodes):
    if not os.path.exists("results"):
        os.mkdir("results")
    with open(file_path, 'w') as file:
        for node, rank in top_100_nodes:
            file.write(f'{node} {rank}\n')

def main():
    graph = read_data(file_path)
    pr = page_rank_sparse(graph, teleport_parameter, tol)
    top_100_nodes = top_nodes(pr, graph.all_nodes)
    write_result('results/sparse_result.txt', top_100_nodes)
    print(f'Top 100 Nodes with their PageRank scores (teleport parameter = {teleport_parameter}):')
    for node, rank in top_100_nodes:
        print(f'NodeID: {node}, PageRank: {rank}')

if __name__ == '__main__':
    main()
