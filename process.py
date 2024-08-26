
import pickle
import numpy as np
from collections import defaultdict

# 数据文件路径
train_data_path = './data/train.txt'
test_data_path = './data/test.txt'
attributes_path = './data/itemAttribute.txt'
index_map_path = './data/node_idx.pkl'

def load_training_data(path, index_map):
    """加载训练数据，转换为内部索引表示，并将评分标准化"""
    user_data, item_data = defaultdict(list), defaultdict(list)
    with open(path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            user_id, count = map(int, line.strip().split('|'))
            for _ in range(count):
                item_id, score = map(float, file.readline().strip().split())
                user_data[user_id].append([index_map[item_id], score / 10])
                item_data[index_map[item_id]].append([user_id, score / 10])
    return user_data, item_data

def load_attributes(path, index_map):
    """加载物品属性数据，并根据索引映射调整。确保每个物品属性正确转换为0或1，并处理None。"""
    attributes = defaultdict(list)
    with open(path, 'r') as file:
        for line in file:
            item_id, attr1, attr2 = line.strip().split('|')
            attr1 = 1 if attr1 != 'None' else 0
            attr2 = 1 if attr2 != 'None' else 0
            if int(item_id) in index_map:
                attributes[index_map[int(item_id)]].extend([attr1, attr2])
    return attributes

def filter_data_by_attributes(user_data, attributes):
    """根据属性过滤数据，生成两个数据集。确保在访问属性前列表有元素。"""
    data_with_attr1, data_with_attr2 = defaultdict(list), defaultdict(list)
    for user_id, items in user_data.items():
        for item_id, score in items:
            if item_id in attributes and len(attributes[item_id]) == 2:
                if attributes[item_id][0]:
                    data_with_attr1[user_id].append([item_id, score])
                if attributes[item_id][1]:
                    data_with_attr2[user_id].append([item_id, score])
    return data_with_attr1, data_with_attr2

def load_test_data(path):
    """加载测试数据集"""
    test_data = defaultdict(list)
    with open(path, 'r') as file:
        for line in file:
            user_id, count = map(int, line.strip().split('|'))
            for _ in range(count):
                item_id = int(file.readline().strip())
                test_data[user_id].append(item_id)
    return test_data

def split_data(data, train_ratio=0.85, shuffle=True):
    """将数据分割为训练集和验证集"""
    train_set, validation_set = defaultdict(list), defaultdict(list)
    for user_id, items in data.items():
        if shuffle:
            np.random.shuffle(items)
        split_point = int(len(items) * train_ratio)
        train_set[user_id], validation_set[user_id] = items[:split_point], items[split_point:]
    return train_set, validation_set

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_data_to_pickle(path, data):
    """将数据保存到Pickle文件"""
    with open(path, 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    print('开始处理数据...')
    with open(index_map_path, 'rb') as file:
        index_map = pickle.load(file)
    user_data, item_data = load_training_data(train_data_path, index_map)
    save_data_to_pickle(train_data_path.replace('.txt', '_user.pkl'), user_data)
    save_data_to_pickle(train_data_path.replace('.txt', '_item.pkl'), item_data)

    attributes = load_attributes(attributes_path, index_map)
    attr1_data, attr2_data = filter_data_by_attributes(user_data, attributes)
    save_data_to_pickle(attributes_path.replace('.txt', '_attr1.pkl'), attr1_data)
    save_data_to_pickle(attributes_path.replace('.txt', '_attr2.pkl'), attr2_data)

    test_data = load_test_data(test_data_path)
    save_data_to_pickle(test_data_path.replace('.txt', '.pkl'), test_data)
    print('数据处理完成！')
