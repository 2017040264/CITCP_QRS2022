
# -*- coding: UTF-8 -*-
"""
@author:cfl
@file:myagent.py
@time:2022/03/10
@software:PyCharm
"""

# 2. agent

# 经验回放数组
import numpy as np
import pickle
from sklearn import neural_network

class ExperienceReplay(object):
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = []
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, experience):
        self.memory.append(experience)

    # 抽样概率是不均等的，新进入经验回放数组的经验，被抽到的概率大。
    def get_batch(self, batch_size=10):

        if len(self.memory) > self.max_memory:
            del self.memory[:len(self.memory) - self.max_memory]

        if batch_size < len(self.memory):
            timerank = range(1, len(self.memory) + 1)
            p = timerank / np.sum(timerank, dtype=float)
            batch_idx = np.random.choice(range(len(self.memory)), replace=False, size=batch_size, p=p)
            batch = [self.memory[idx] for idx in batch_idx]
        else:
            batch = self.memory

        return batch


# 这里是基础的agent，相当于父类，之后的agents都是它的子类
class BaseAgent(object):
    # 基于历史信息
    # 传入历史信息的长度
    def __init__(self, histlen):
        self.single_testcases = True
        self.train_mode = True
        self.histlen = histlen

    def get_action(self, s):
        return 0

    def get_all_actions(self, states):
        """ Returns list of actions for all states """
        return [self.get_action(s) for s in states]

    def reward(self, reward):
        pass

    def save(self, filename):
        """ Stores agent as pickled file """
        # 2表示新的二进制协议
        pickle.dump(self, open(filename + '.p', 'wb'), 2)


    # 使用classmethod修饰，不需要实例化类也可以调用这个函数，但是需要表示自身类的cls参数
    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename + '.p', 'rb'))


class NetworkAgent(BaseAgent):

    # state_size 状态空间的大小
    # action_size 动作空间的大小
    # hidden_size MLP的参数
    # histlen 历史长度
    def __init__(self, action_size, hidden_size, histlen,name):
        super(NetworkAgent, self).__init__(histlen=histlen)
        self.name = name
        self.experience_length = 10000
        self.experience_batch_size = 1000  # 每次从经验数组中取出的经验数量
        self.experience = ExperienceReplay(max_memory=self.experience_length)
        self.episode_history = []  # 执行历史
        self.iteration_counter = 0
        self.action_size = action_size

        if isinstance(hidden_size, tuple):
            self.hidden_size = hidden_size
        else:
            self.hidden_size = (hidden_size,)

        self.model = None
        self.model_fit = False
        self.init_model(True)

    # TODO This could improve performance (if necessary)
    # def get_all_actions(self, states):
    #   try:

    def init_model(self, warm_start=True):
        if self.action_size == 1:
            # 多层感知机分类器
            # max_iter: 最大迭代次数。 对于Adam，表示 ：how many times each data point will be used

            # warm_start:When set to True, reuse the solution of the previous call to fit as initialization,
            # otherwise, just erase the previous solution.

            self.model = neural_network.MLPClassifier(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                      warm_start=warm_start, solver='adam', max_iter=750)
        else:
            # 多层感知机回归器
            self.model = neural_network.MLPRegressor(hidden_layer_sizes=self.hidden_size, activation='relu',
                                                     warm_start=warm_start, solver='adam', max_iter=750)
        self.model_fit = False

    def get_action(self, s):
        if self.model_fit:
            if self.action_size == 1:
                a = self.model.predict_proba(np.array(s).reshape(1, -1))[0][1]
            else:
                a = self.model.predict(np.array(s).reshape(1, -1))[0]
        else:
            a = np.random.random()

        # trian_mode 在父类中
        if self.train_mode:
            self.episode_history.append((s, a))

        return a

    def reward(self, rewards):

        # 如果不是在训练模型
        if not self.train_mode:
            return

        try:
            # rewards 是 string 或者 number
            x = float(rewards)
            # * 是做重复运算
            rewards = [x] * len(self.episode_history)
        except:
            if len(rewards) < len(self.episode_history):
                raise Exception('Too few rewards')

        self.iteration_counter += 1

        # 首先 len(episode_history)==rewards ??  yes，都是遍历同一个sc 得到

        # 其次 (state, action), reward 是否可以一一对应?? yes,对于sc中的每条测试用例，首先preprocess函数
        # （netwaork agent 是执行preprocess_continuous），然后得到一个输出，作为get_action的输入，对应一个（state,action）,
        # 并返回action,这个action是对优先级的预测;对sc遍历一遍后，得到了完整的episode_history,然后进行一个submit()操作，
        # 得到[detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]，然后
        # 对每条测试用例计算奖励。 因此，可以认为(state, action), reward 是一一对应的。

        for ((state, action), reward) in zip(self.episode_history, rewards):
            self.experience.remember((state, reward))

        self.episode_history = []

        # 每5个周期训练一次网络
        if self.iteration_counter == 1 or self.iteration_counter % 5 == 0:
            self.learn_from_experience()

    def learn_from_experience(self):
        experiences = self.experience.get_batch(self.experience_batch_size)
        x, y = zip(*experiences)

        if self.model_fit:
            try:
                # Update the model with a single iteration over the given data.
                self.model.partial_fit(x, y)
            except ValueError:
                self.init_model(warm_start=False)

                # Fit the model to data matrix X and target(s) y.
                # y: The target values (class labels in classification, real numbers in regression).
                self.model.fit(x, y)

                self.model_fit = True
        else:
            self.model.fit(x, y)  # Call fit once to learn classes
            self.model_fit = True

