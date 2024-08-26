import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from process import split_data, load_pkl, save_data_to_pickle
import time
import heapq
test_pkl = './data/test.pkl'
bx_pkl = './data/bx.pkl'
bi_pkl = './data/bi.pkl'
idx_pkl = './data/node_idx.pkl'


# 带有偏置和正则化的矩阵分解模型
class SVD:
    def __init__(self, model_path='./model',
                 data_path='./data/train_user.pkl', lr=5e-3,
                 lamda1=1e-2, lamda2=1e-2, lamda3=1e-2, lamda4=1e-2,
                 factor=50, K=10):
        self.K = K  # 保存一个实例变量
        self.num_users = 19835  # 用户数量
        self.num_items = 455705  # 物品数量
        self.bx = load_pkl(bx_pkl)  # 用户偏置
        self.bi = load_pkl(bi_pkl)  # 物品偏置
        self.lr = lr  # 学习率
        self.lamda1 = lamda1  # 正则化系数，乘以P
        self.lamda2 = lamda2  # 正则化系数，乘以Q
        self.lamda3 = lamda3  # 正则化系数，乘以bx
        self.lamda4 = lamda4  # 正则化系数，乘以bi
        self.factor = factor  # 隐向量维度
        self.idx = load_pkl(idx_pkl)
        self.train_user_data = load_pkl(data_path)
        self.train_data, self.valid_data = split_data(self.train_user_data)
        self.test_data = load_pkl(test_pkl)
        self.globalmean = self.get_globalmean()  # 全局平均分
        # 随机初始化矩阵Q(物品)和P(用户)
        self.Q = np.random.normal(0, 0.1, (self.factor, len(self.bi)))
        self.P = np.random.normal(0, 0.1, (self.factor, len(self.bx)))
        self.model_path = model_path

    def get_globalmean(self):
        score_sum, count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                score_sum += score
                count += 1
        return score_sum / count

    def predict(self, user_id, item_id):
        pre_score = self.globalmean + \
            self.bx[user_id] + \
            self.bi[item_id] + \
            np.dot(self.P[:, user_id], self.Q[:, item_id])
        return pre_score

    def loss(self, is_valid=False):
        loss, count = 0.0, 0
        data = self.valid_data if is_valid else self.train_data
        for user_id, items in data.items():
            for item_id, score in items:
                loss += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        # 如果是训练集，计算正则化项
        if not is_valid:
            loss += self.lamda1 * np.sum(self.P ** 2)
            loss += self.lamda2 * np.sum(self.Q ** 2)
            loss += self.lamda3 * np.sum(self.bx ** 2)
            loss += self.lamda4 * np.sum(self.bi ** 2)
        loss /= count
        return loss

    def precision_recall(self, k):
        """
        :param k: 推荐列表的长度
        :return: 精确率和召回率
        """
        hits = 0
        rec_count = 0
        test_count = 0
        for user_id in self.test_data.keys():
            test_items = self.test_data[user_id]
            scores = {item_id: score for item_id, score in self.train_user_data[user_id]}
            rec_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            rec_items = [item[0] for item in rec_items]
            for item in rec_items:
                if item in test_items:
                    hits += 1
            rec_count += len(rec_items)
            test_count += len(test_items)

        precision = hits / float(rec_count) if rec_count > 0 else 0
        recall = hits / float(test_count) if test_count > 0 else 0
        return precision, recall

    def rmse(self):
        rmse, count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                rmse += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        rmse /= count
        rmse = np.sqrt(rmse)
        return rmse

    def train(self, epochs=10, save=False, load=False):
        if load:
            self.load_weight()
        print('开始训练...')
        # 初始化记录loss和rmse的列表
        train_losses = []
        valid_losses = []
        rmses = []
        for epoch in range(epochs):
            for user_id, items in tqdm(self.train_data.items(), desc=f'第 {epoch + 1} 轮'):
                for item_id, score in items:
                    error = score - self.predict(user_id, item_id)
                    grad_P = error * self.Q[:, item_id] - self.lamda1 * self.P[:, user_id]
                    grad_Q = error * self.P[:, user_id] - self.lamda2 * self.Q[:, item_id]
                    grad_bx = error - self.lamda3 * self.bx[user_id]
                    grad_bi = error - self.lamda4 * self.bi[item_id]
                    self.P[:, user_id] += self.lr * grad_P
                    self.Q[:, item_id] += self.lr * grad_Q
                    self.bx[user_id] += self.lr * grad_bx
                    self.bi[item_id] += self.lr * grad_bi
            train_loss = self.loss()
            valid_loss = self.loss(is_valid=True)
            precision, recall = self.precision_recall(k=10)
            f1 = self.f1_score(k=10)
            rmse = self.rmse()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            rmses.append(rmse)
            print(f'第 {epoch + 1} 轮 训练损失: {train_loss:.6f}, 验证损失: {valid_loss:.6f}, RMSE: {rmse:.6f}')
            print(f'精确率: {precision:.6f}, 召回率: {recall:.6f}, F1分数: {f1:.6f}')
        print('训练完成。')

        if save:
            self.save_weight()

        # 绘制loss曲线
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(range(len(valid_losses)), valid_losses, label='Valid Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # 绘制rmse曲线
        plt.figure()
        plt.plot(range(len(rmses)), rmses)
        plt.title('RMSE over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.show()



        return train_losses, valid_losses, rmses

    def f1_score(self, k):
        precision, recall = self.precision_recall(k)
        if precision + recall == 0:
            return 0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def test(self, write_path='./result/result.txt', load=True):
        if load:
            self.load_weight()
        print('开始测试...')
        predict_score = defaultdict(list)
        precision, recall = self.precision_recall(k=10)
        f1 = self.f1_score(k=10)
        print(f'精确率: {precision:.6f}, 召回率: {recall:.6f}, F1分数: {f1:.6f}')
        for user_id, item_list in self.test_data.items():
            for item_id in item_list:
                if item_id not in self.idx:
                    pre_score = self.globalmean * 10
                else:
                    new_id = self.idx[item_id]
                    pre_score = self.predict(user_id, new_id) * 10
                    if pre_score > 100.0:
                        pre_score = 100.0
                    elif pre_score < 0.0:
                        pre_score = 0.0

                predict_score[user_id].append((item_id, pre_score))
        print('测试完成。')

        def write_result(predict_score, write_path):
            print('开始写入结果...')
            with open(write_path, 'w') as f:
                for user_id, items in predict_score.items():
                    f.write(f'{user_id}|6\n')
                    for item_id, score in items:
                        f.write(f'{item_id} {score}\n')
            print('写入完成。')

        if write_path:
            write_result(predict_score, write_path)
        return predict_score

    def load_weight(self):
        print('加载模型权重...')
        self.bx = load_pkl(self.model_path + '/bx.pkl')
        self.bi = load_pkl(self.model_path + '/bi.pkl')
        self.P = load_pkl(self.model_path + '/P.pkl')
        self.Q = load_pkl(self.model_path + '/Q.pkl')
        print('加载完成。')

    def save_weight(self):
        print('保存模型权重...')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        save_data_to_pickle(self.model_path + '/bx.pkl', self.bx)
        save_data_to_pickle(self.model_path + '/bi.pkl', self.bi)
        save_data_to_pickle(self.model_path + '/P.pkl', self.P)
        save_data_to_pickle(self.model_path + '/Q.pkl', self.Q)
        print('保存完成。')

import matplotlib.pyplot as plt
if __name__ == '__main__':
    svd = SVD()

    train_losses, valid_losses, rmses = svd.train(epochs=10, save=True, load=True)

    svd.test(write_path='./result/svd1.txt')
    rmse = svd.rmse()
    print(f'最终的均方根误差（RMSE）: {rmse:.6f}')

