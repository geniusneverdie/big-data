
import pandas as pd

train_path = './data/train.txt'

def extract_statistics(file_path):
    """
    从数据文件中提取统计信息，包括用户数、物品数、评分数等。
    :param file_path: 数据文件的路径
    """
    data = []
    with open(file_path, 'r') as file:
        while (line := file.readline()) != '':
            user_id, num_ratings = map(int, line.strip().split('|'))
            for _ in range(num_ratings):
                line = file.readline()
                item_id, score = map(float, line.strip().split())
                data.append([user_id, item_id, score])

    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'score'])

    user_count = df['user_id'].nunique()
    item_count = df['item_id'].nunique()
    total_ratings = len(df)
    max_user_id = df['user_id'].max()
    max_item_id = df['item_id'].max()
    average_rating = df['score'].mean()

    print("用户数:", user_count)
    print("评分的物品数:", item_count)
    print("评分总数:", total_ratings)
    print("最大用户ID:", max_user_id)
    print("最大物品ID:", max_item_id)
    print("平均评分:", average_rating)


if __name__ == '__main__':
    extract_statistics(train_path)
