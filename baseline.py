
import numpy as np
from collections import defaultdict
from process import load_pkl

# 定义数据路径
test_pkl = './data/test.pkl'
bx_pkl = './data/bx.pkl'
bi_pkl = './data/bi.pkl'
idx_pkl = './data/node_idx.pkl'
train_user_pkl = './data/train_user.pkl'

class Baseline:
    def __init__(self):
        # 使用load_pkl来加载数据
        self.bx = load_pkl(bx_pkl)  # 用户偏置
        self.bi = load_pkl(bi_pkl)  # 物品偏置
        self.idx = load_pkl(idx_pkl)  # 索引映射
        self.train_user_data = load_pkl(train_user_pkl)  # 训练数据
        self.test_data = load_pkl(test_pkl)  # 测试数据
        self.globalmean = self.get_globalmean()  # 计算全局平均分

    def get_globalmean(self):
        """计算全局平均评分"""
        score_sum = sum(score for items in self.train_user_data.values() for _, score in items)
        count = sum(len(items) for items in self.train_user_data.values())
        return score_sum / count if count else 0

    def predict(self, user_id, item_id):
        """根据用户偏置、物品偏置和全局平均值预测评分"""
        user_bias = self.bx[user_id] if user_id < len(self.bx) else 0
        item_bias = self.bi[item_id] if item_id < len(self.bi) else 0
        pre_score = self.globalmean + user_bias + item_bias
        return pre_score

    def rmse(self):
        """计算均方根误差 (Root Mean Square Error, RMSE)"""
        scores = [score - self.predict(user_id, item_id) for user_id, items in self.train_user_data.items() for item_id, score in items]
        return np.sqrt(np.mean(np.square(scores)))

    def test(self, write_path='./result/result.txt'):
        """进行测试并将结果写入文件"""
        print('开始测试...')
        predict_score = defaultdict(list)
        for user_id, item_list in self.test_data.items():
            for item_id in item_list:
                adjusted_item_id = self.idx.get(item_id, item_id)  # 获取调整后的item_id
                if adjusted_item_id >= len(self.bi):
                    pre_score = self.globalmean * 10
                else:
                    pre_score = self.predict(user_id, adjusted_item_id) * 10
                pre_score = np.clip(pre_score, 0, 100)  # 限制预测值在0到100之间
                predict_score[user_id].append((item_id, pre_score))
        print('测试完成。')

        if write_path:
            self.write_result(predict_score, write_path)
        return predict_score

    def write_result(self, predict_scores, write_path):
        """将预测结果写入文件"""
        print('开始写入结果...')
        with open(write_path, 'w') as f:
            for user_id, items in predict_scores.items():
                f.write(f'{user_id}|{len(items)}\n')  # 写入用户ID和条目数
                for item_id, score in items:
                    f.write(f'{item_id} {score}\n')
        print('写入完成。')

if __name__ == '__main__':
    baseline = Baseline()
    baseline.test(write_path='./result/baseline.txt')
    rmse = baseline.rmse()
    print(f'RMSE: {rmse:.6f}')
