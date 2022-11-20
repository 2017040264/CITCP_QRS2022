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
    rank_idx = np.array(result[-1]) - 1  # 执行过且检测到错误的测试用例的索引
    # scheduled_testcases 是执行过的测试用例列表，一条测试用例信息是一个元素
    no_scheduled = len(sc.scheduled_testcases)

    rewards = np.zeros(no_scheduled)
    rewards[rank_idx] = 2

    ordered_rewards = []

    for tc in sc.testcases():
        try:
            idx = sc.scheduled_testcases.index(tc)
            ordered_rewards.append(rewards[idx])
        except ValueError:
            ordered_rewards.append(0.0)  # Unscheduled test case

    return ordered_rewards


def NV_ALL(sc, bonus):
    novlty_rewards = []
    for tc in sc.testcases():
        hisresults = tc['LastResults']
        novlty = 0
        if len(hisresults) == 0:  # novlty
            novlty = bonus

        novlty_rewards.append(novlty)
    return novlty_rewards


# 附加奖励为3，将新出现的看做比失败的更重要
# 附加奖励为2，将新出现的看做与失败的同等重要
# 附加奖励为1，将新出现的看做不如失败的重要


# 附加奖励为3，奖励不归一化
def add_tcfail_3(result, sc):
    ordered_rewards = tcfail(result, sc)
    bonus = NV_ALL(sc=sc, bonus=3)  # 所有测试用力的额外奖励

    try:
        x = int(ordered_rewards)
        for i in range(len(bonus)):
            bonus[i] = x + bonus[i]
    except:
        for i in range(len(bonus)):
            bonus[i] = ordered_rewards[i] + bonus[i]

    return bonus


# 附加奖励为2，奖励不归一化
def add_tcfail_2(result, sc):
    ordered_rewards = tcfail(result, sc)
    bonus = NV_ALL(sc=sc, bonus=2)  # 所有测试用力的额外奖励

    try:
        x = int(ordered_rewards)
        for i in range(len(bonus)):
            bonus[i] = x + bonus[i]
    except:
        for i in range(len(bonus)):
            bonus[i] = ordered_rewards[i] + bonus[i]

    return bonus


# 附加奖励为1，奖励不归一化
def add_tcfail_1(result, sc):
    ordered_rewards = tcfail(result, sc)
    bonus = NV_ALL(sc=sc, bonus=1)  # 所有测试用力的额外奖励

    try:
        x = int(ordered_rewards)
        for i in range(len(bonus)):
            bonus[i] = x + bonus[i]
    except:
        for i in range(len(bonus)):
            bonus[i] = ordered_rewards[i] + bonus[i]

    return bonus


def failcount(result, sc=None):
    return float(result[0])


def timerank(result, sc):
    if result[0] == 0:
        return 0.0

    total = result[0]
    rank_idx = np.array(result[-1]) - 1
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
