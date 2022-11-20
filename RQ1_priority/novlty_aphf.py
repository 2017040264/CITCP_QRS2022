#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:cfl
@file:novlty_aphf.py
@time:2022/03/10
@software:PyCharm
"""

from __future__ import division
from datetime import datetime
import numpy as np
import os
import time
import multiprocessing
import pickle
import warnings


from myagent import NetworkAgent, ExperienceReplay
from myenv import get_scenario

from myreward import tcfail

from myvisualization import visualize


DEFAULT_NO_SCENARIOS = 1000
DEFAULT_NO_ACTIONS = 100
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EPSILON = 0.2
DEFAULT_DUMP_INTERVAL = 100
DEFAULT_VALIDATION_INTERVAL = 100
DEFAULT_PRINT_LOG = False
DEFAULT_PLOT_GRAPHS = False
DEFAULT_NO_HIDDEN_NODES = 12

DEFAULT_TODAY = datetime.today()
print('DEFAULT_TODAY :',DEFAULT_TODAY)

ITERATIONS = 30  # Number of times the experiment is repeated, 原始是30，这里先修改的小一点
CI_CYCLES = 1000

USE_LATEX=False
DATA_DIR = 'results'
FIGURE_DIR = 'results_csv'
#PARALLEL = False
PARALLEL = True
PARALLEL_POOL_SIZE = 4

RUN_EXPERIMENT = True
VISUALIZE_RESULTS = True


# state是包含以下内容的一行数据 ['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
def preprocess_continuous(state, scenario_metadata, histlen):
    #   'minExecTime': min(execTimes), execTimes是LastRun
    #   'maxExecTime': max(execTimes), execTimes是LastRun
    if scenario_metadata['maxExecTime'] > scenario_metadata['minExecTime']:
        # 该测试用例到该循环周期结束 的时间
        # 该循环周期的总时间
        time_since = (scenario_metadata['maxExecTime'] - state['LastRun']).total_seconds() / (
            scenario_metadata['maxExecTime'] - scenario_metadata['minExecTime']).total_seconds()
    else:
        time_since = 0

    # [0:histlen]的历史执行信息,但是如果len(state['LastResults])<histlen,那么len(histroy)<histlen
    history = [1 if res else 0 for res in state['LastResults'][0:histlen]]

    # 类似于填充操作，将history填充至 histlen长度
    if len(history) < histlen:
        history.extend([1] * (histlen - len(history)))

    row = [
        state['Duration'] / scenario_metadata['totalTime'],
        time_since
    ]
    row.extend(history)

    return tuple(row)  # 形如 (0.5, 0.3, 1, 1, 0)


def process_scenario(agent, sc, preprocess):
    scenario_metadata = sc.get_ta_metadata()

    if agent.single_testcases:
        for row in sc.testcases():
            # Build input vector: preprocess the observation
            # row  is a single testcase  ['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
            #   对于sc中的每条测试用例，首先preprocess函数
            # （netwaork agent 是执行preprocess_continuous），然后得到一个输出（state），作为get_action的输入，对应一个（state,action）,
            # 并返回action,这个action是对优先级的预测;
            x = preprocess(row, scenario_metadata, agent.histlen)
            action = agent.get_action(x)
            row['CalcPrio'] = action  # Store prioritization

    else:
        states = [preprocess(row, scenario_metadata, agent.histlen) for row in sc.testcases()]
        actions = agent.get_all_actions(states)

        for (tc_idx, action) in enumerate(actions):
            sc.set_testcase_prio(action, tc_idx)

    # Submit prioritized file for evaluation
    # step the environment and get new measurements
    return sc.submit()


def process_scenario_new(agent, sc, preprocess):
    scenario_metadata = sc.get_ta_metadata()

    if agent.single_testcases:
        for row in sc.testcases():
            # Build input vector: preprocess the observation
            # row  is a single testcase  ['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
            #   对于sc中的每条测试用例，首先preprocess函数
            # （netwaork agent 是执行preprocess_continuous），然后得到一个输出（state），作为get_action的输入，对应一个（state,action）,
            # 并返回action,这个action是对优先级的预测;
            x = preprocess(row, scenario_metadata, agent.histlen)

            if len(row["LastResults"])==0:  # 新出现的测试用例，优先级直接设为1
                action=1.0
                agent.episode_history.append((x, action))
            else:
                action = agent.get_action(x)
            #print("优先级：",action)
            row['CalcPrio'] = action  # Store prioritization

    else:
        states = [preprocess(row, scenario_metadata, agent.histlen) for row in sc.testcases()]
        actions = agent.get_all_actions(states)

        for (tc_idx, action) in enumerate(actions):
            sc.set_testcase_prio(action, tc_idx)

    # Submit prioritized file for evaluation
    # step the environment and get new measurements
    return sc.submit()

class PrioLearning(object):
    # DEFAULT_DUMP_INTERVAL = 100
    # DEFAULT_VALIDATION_INTERVAL = 100
    # file_prefix： 文件前缀名
    # interval: 间隔
    def __init__(self, agent, scenario_provider, file_prefix, reward_function, output_dir, preprocess_function,
                 dump_interval=DEFAULT_DUMP_INTERVAL, validation_interval=DEFAULT_VALIDATION_INTERVAL):
        self.agent = agent
        self.scenario_provider = scenario_provider
        self.reward_function = reward_function
        self.preprocess_function = preprocess_function
        self.replay_memory = ExperienceReplay()
        self.validation_res = []

        self.dump_interval = dump_interval
        self.validation_interval = validation_interval

        self.today = DEFAULT_TODAY

        # 3种文件的存储文件名称
        self.file_prefix = file_prefix
        self.val_file = os.path.join(output_dir, '%s_val' % file_prefix)
        self.stats_file = os.path.join(output_dir, '%s_stats' % file_prefix)
        self.agent_file = os.path.join(output_dir, '%s_agent' % file_prefix)


    # sc是一个虚拟场景，包含一个CI周期的数据
    def process_scenario(self, sc):

        # 对于network agent来说，self.preprocess_function是调用的 preprocess_continuous 函数，连续数据
        # result 包含以下内容 [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]
        if self.agent.name=="mlp_cla_new":
            result = process_scenario_new(self.agent, sc, self.preprocess_function)
        elif self.agent.name=="mlp_cla_old":
            result = process_scenario(self.agent, sc, self.preprocess_function)
        else:
            print("self.agent.name error!")

        # reward 是对每个测试用例的奖励,list
        reward = self.reward_function(result, sc)

        self.agent.reward(reward)

        return result, reward

    def replay_experience(self, batch_size):
        batch = self.replay_memory.get_batch(batch_size)

        for sc in batch:
            (result, reward,apr,nor) = self.process_scenario(sc)
            print('Replay Experience: %s / %.2f' % (result, np.mean(reward)))

    # 奖励值的方差
    def comput2(self,lista):
        ave=np.mean(lista)
        newl=[]
        for i in range(len(lista)):
            newl.append((lista[i]-ave)**2)
        return np.mean(newl)

    def train(self):
        #print(" 奖励函数的名字：", self.reward_function.__name__)
        stats = {
            'agent': self.agent.name,
            'scenarios': [],
            'rewards': [],
            'rewards_variance':[],
            'durations': [],
            'detected': [],
            'missed': [],
            'ttf': [],
            'napfd': [],
            'recall': [],
            'avg_precision': [],
            'result': [],
            'step': [],
            'env': self.scenario_provider.name,
            'rewardfun': self.reward_function.__name__,
        }

        # 'action_size': self.agent.action_size,
        # 'history_length': self.agent.histlen,
        # 'sched_time': self.scenario_provider.avail_time_ratio,
        # 'hidden_size': 'x'.join(str(x) for x in self.agent.hidden_size) if hasattr(self.agent, 'hidden_size')else 0
        # sum_actions = 0
        # sum_detected = 0
        # sum_missed = 0
        # sum_reward = 0
        sum_scenarios = 0

        # 调用 self.scenario_provider 中的get函数
        # 每个CI周期的数据组成一个sc
        for (i, sc) in enumerate(self.scenario_provider, start=1):
            # 最大循环限制
            #             if i > no_scenarios:
            #                 break

            start = time.time()

            # result : [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]
            # reward 是每个测试用例的奖励
            (result, reward) = self.process_scenario(sc)
            #print(reward,aphf_rewards,novlty_rewards)
            end = time.time()

            # Statistics
            # sum_detected += result[0]
            # sum_missed += result[1]
            # sum_reward += np.mean(reward)
            # sum_actions += 1
            sum_scenarios += 1
            duration = end - start

            stats['scenarios'].append(sc.name)
            stats['rewards'].append(np.mean(reward))
            stats['rewards_variance'].append(self.comput2(reward))
            stats['durations'].append(duration)
            stats['detected'].append(result[0])
            stats['missed'].append(result[1])
            stats['ttf'].append(result[2])
            stats['napfd'].append(result[3])
            stats['recall'].append(result[4])
            stats['avg_precision'].append(result[5])
            stats['result'].append(result)
            stats['step'].append(sum_scenarios)


            # wb方式会覆盖以前的信息，这里是不是没必要？
            # Data Dumping
            if self.dump_interval > 0 and sum_scenarios % self.dump_interval == 0:
                pickle.dump(stats, open(self.stats_file + '.p', 'wb'))

        # end for  遍历完一个数据集
        if self.dump_interval > 0:
            # self.agent.save(self.agent_file)
            pickle.dump(stats, open(self.stats_file + '.p', 'wb'))

        # 返回整个数据集的napfd均值
        return np.mean(stats['napfd'])


def exp_run_industrial_datasets(iteration):
    ags = [
        lambda: (NetworkAgent(histlen=DEFAULT_HISTORY_LENGTH,
                              action_size=1,
                              hidden_size=DEFAULT_NO_HIDDEN_NODES,
                              name="mlp_cla_new"),
                 preprocess_continuous),


        lambda: (NetworkAgent(histlen=DEFAULT_HISTORY_LENGTH,
                              action_size=1,
                              hidden_size=DEFAULT_NO_HIDDEN_NODES,
                              name="mlp_cla_old"),
                 preprocess_continuous)]

    #datasets = ['apache_hive','apache_drill','apache_commons',"apache_parquet","paintcontrol",'iofrol','dspace','google_auto','apache_tajo','google_closure','google_guava','mybatis','rails']
    datasets = ['apache_hive' ]

    reward_funs = {
        "tcfail":tcfail,
    }


    avg_napfd = []

    for i, get_agent in enumerate(ags):
        for sc in datasets:
            print('正在执行：iteration = {},dataset = {}'.format(iteration, sc))
            for (reward_name, reward_fun) in reward_funs.items():

                agent, preprocessor = get_agent()
                file_appendix = 'rq_%s_%s_%s_%d' % (agent.name, sc, reward_name, iteration)

                scenario = get_scenario(sc)

                rl_learning = PrioLearning(agent=agent,
                                           scenario_provider=scenario,
                                           reward_function=reward_fun,
                                           preprocess_function=preprocessor,
                                           file_prefix=file_appendix,
                                           dump_interval=100,
                                           validation_interval=0,
                                           output_dir=DATA_DIR)

                res = rl_learning.train()
                avg_napfd.append(res)

    return avg_napfd


def run_experiments(exp_fun, parallel=PARALLEL):
    if parallel:
        p = multiprocessing.Pool(PARALLEL_POOL_SIZE)
        avg_res = p.map(exp_fun, range(ITERATIONS))
    else:
        avg_res = [exp_fun(i) for i in range(ITERATIONS)]

    print('Ran experiments: %d results' % len(avg_res))




def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    warnings.filterwarnings('ignore')
    b_time = time.time()
    run_experiments(exp_run_industrial_datasets, parallel=PARALLEL)
    print("执行时间：{:.3f}s".format(time.time() - b_time))

    visualize(p='RETECS_tcfail')
    print("all is Over")


if __name__ == '__main__':
    main()
