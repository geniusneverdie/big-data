

import pickle
import numpy as np

# 数据文件路径
train_user_pkl = './data/train_user.pkl'
train_item_pkl = './data/train_item.pkl'

# 预定义的用户和物品数量
user_num = 19835
item_num = 455691
ratings_num = 5001507

def calculate_bias(train_data_user, train_data_item):
    """
    计算并返回全局评分均值、用户偏差和物品偏差。
    :param train_data_user: 用户到(物品,评分)列表的映射字典
    :param train_data_item: 物品到(用户,评分)列表的映射字典
    """
    global_total = sum(score for items in train_data_user.values() for _, score in items)
    global_average = global_total / ratings_num

    user_bias = np.zeros(user_num, dtype=np.float64)
    item_bias = np.zeros(item_num, dtype=np.float64)

    # 计算用户偏差
    for user_id, items in train_data_user.items():
        total_score = sum(score for _, score in items)
        user_bias[user_id] = total_score / len(items) - global_average

    # 计算物品偏差
    for item_id, users in train_data_item.items():
        total_score = sum(score for _, score in users)
        item_bias[item_id] = total_score / len(users) - global_average

    return global_average, user_bias, item_bias

if __name__ == '__main__':
    print('正在加载数据...')
    with open(train_user_pkl, 'rb') as file:
        train_user_data = pickle.load(file)
    with open(train_item_pkl, 'rb') as file:
        train_item_data = pickle.load(file)
    print('数据加载完成。')

    # 计算偏差
    average_rating, user_bias, item_bias = calculate_bias(train_user_data, train_item_data)

    print('正在保存数据...')
    with open('./data/bx.pkl', 'wb') as file:
        pickle.dump(user_bias, file)
    with open('./data/bi.pkl', 'wb') as file:
        pickle.dump(item_bias, file)
    print('数据保存完成。')

    print('全局评分均值：', average_rating)
