
# -*- coding: UTF-8 -*-
"""
@author:cfl
@file:myreward.py
@time:2022/03/10
@software:PyCharm
"""

import numpy as np

def tcfail(result, sc):
    # result 包含以下内容 [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]
    # sc 是一个循环周期的数据，['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
    if result[0] == 0:
        return 0.0

    total = result[0]
    rank_idx = np.array(result[-1])-1 # 执行过且检测到错误的测试用例的索引
    # scheduled_testcases 是执行过的测试用例列表，一条测试用例信息是一个元素
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1

    ordered_rewards = []

    for tc in sc.testcases():
        try:
            idx = sc.scheduled_testcases.index(tc)
            ordered_rewards.append(rewards[idx])
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case

    return ordered_rewards


def failcount(result, sc=None):
    return float(result[0])

def timerank(result, sc):
    if result[0] == 0:
        return 0.0

    total = result[0]
    rank_idx = np.array(result[-1])-1
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 1
    rewards = np.cumsum(rewards)  # Rewards for passed testcases
    rewards[rank_idx] = total  # Rewards for failed testcases

    ordered_rewards = []

    for tc in sc.testcases():
        try:
            idx = sc.scheduled_testcases.index(tc)  # Slow call
            ordered_rewards.append(rewards[idx])
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case

    return ordered_rewards