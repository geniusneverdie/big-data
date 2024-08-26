
import pickle

# 设置训练数据文件的路径
train_path = './data/train.txt'

def create_index(train_path):
    """
    从训练数据文件中提取所有唯一的物品ID，并为每个ID分配一个唯一的索引。
    :param train_path: 训练数据文件路径
    :return: 字典，映射物品ID到唯一索引
    """
    unique_items = set()
    with open(train_path, 'r') as file:
        while (line := file.readline()) != '':
            _, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = file.readline()
                item_id, _ = map(int, line.strip().split())
                unique_items.add(item_id)

    # 创建物品ID到索引的映射字典
    item_to_index = {item: idx for idx, item in enumerate(sorted(unique_items))}
    return item_to_index

if __name__ == '__main__':
    item_index = create_index(train_path)
    with open('./data/node_idx.pkl', 'wb') as file:
        pickle.dump(item_index, file)
    print('索引创建完成。')

