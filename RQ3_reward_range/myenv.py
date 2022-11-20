
# -*- coding: UTF-8 -*-
"""
@author:cfl
@file:myenv.py
@time:2022/03/10
@software:PyCharm
"""

from datetime import datetime, timedelta
import random
import os
import json
import pandas as pd

# 1. the env
class VirtualScenario(object):
    # 虚拟场景
    # available_time 是 测试用例可执行时间（已经取0.5倍）
    # testcases 当前循环周期中的数据，内容包含['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults']
    # solutions 是当前循环周期的 {Id：verdict} 字典
    def __init__(self, available_time, testcases=[], solutions={}, name_suffix='vrt', schedule_date=datetime.today()):
        self.available_time = available_time
        self.gen_testcases = testcases
        self.solutions = solutions
        self.no_testcases = len(testcases)
        self.name = name_suffix
        self.scheduled_testcases = []  # 该循环中执行过的测试用例
        self.schedule_date = schedule_date

    def testcases(self):
        return iter(self.gen_testcases)

    def submit(self):
        # Sort tc by Prio ASC (for backwards scheduling), break ties randomly
        # 'CalcPrio'是测试用例优先级
        # random.random()是生成随机数，在0至1的范围之内
        sorted_tc = sorted(self.gen_testcases, key=lambda x: (x['CalcPrio'], random.random()))

        # Build prefix sum of durations to find cut off point
        scheduled_time = 0   # 执行的时间
        detection_ranks = [] # 记录错误的测试用例在当前测试套件中的位置
        undetected_failures = 0
        rank_counter = 1  #

        while sorted_tc:
            # pop()是移除最后一个元素，也是列表中优先级最高的元素
            cur_tc = sorted_tc.pop()

            if scheduled_time + cur_tc['Duration'] <= self.available_time:
                # 如果值为1 ，表明是存在错误的测试用例
                if self.solutions[cur_tc['Id']]:
                    detection_ranks.append(rank_counter)

                scheduled_time += cur_tc['Duration']
                self.scheduled_testcases.append(cur_tc)
                rank_counter += 1

            else:
                undetected_failures += self.solutions[cur_tc['Id']]

        detected_failures = len(detection_ranks)
        total_failure_count = sum(self.solutions.values())

        # 断言机制，确保检测到的失败用例和未检测到的失败用例总和正确
        assert undetected_failures + detected_failures == total_failure_count

        if total_failure_count > 0:
            # 检测到的第一个错误的在该测试套件中的位置
            ttf = detection_ranks[0] if detection_ranks else 0

            if undetected_failures > 0:
                p = (detected_failures / total_failure_count)
            else:
                p = 1

            napfd = p - sum(detection_ranks) / (total_failure_count * self.no_testcases) + p / (2 * self.no_testcases)
            recall = detected_failures / total_failure_count
            avg_precision = 123
        else:
            ttf = 0
            napfd = 1
            recall = 1
            avg_precision = 1

        return [detected_failures, undetected_failures, ttf, napfd, recall, avg_precision, detection_ranks]

    def get_ta_metadata(self):
        execTimes, durations = zip(*[(tc['LastRun'], tc['Duration']) for tc in self.testcases()])

        metadata = {
            'availAgents': 1,
            'totalTime': self.available_time,
            'minExecTime': min(execTimes),
            'maxExecTime': max(execTimes),
            'scheduleDate': self.schedule_date,
            'minDuration': min(durations),
            'maxDuration': max(durations)
        }

        return metadata

    def reduce_to_schedule(self):
        """ Creates a new scenario consisting of all scheduled test cases and their outcomes (for replaying) """
        #  self.scheduled_testcases = []   该循环中执行过的测试用例
        # scheduled_time 是 执行过的测试用例的总时间，<= available_time
        scheduled_time = sum([tc['Duration'] for tc in self.scheduled_testcases])

        # 所有测试用例的执行时间，是 available_time的2倍
        total_time = sum([tc['Duration'] for tc in self.testcases()])

        available_time = self.available_time * scheduled_time / total_time

        # 该循环周期中执行过的测试用例的的{id： verdict}字典
        solutions = {tc['Id']: self.solutions[tc['Id']] for tc in self.scheduled_testcases}

        return VirtualScenario(available_time, self.scheduled_testcases, solutions, self.name, self.schedule_date)


    def set_testcase_prio(self, prio, tcid=-1):
        self.gen_testcases[tcid]['CalcPrio'] = prio


    def clean(self):
        for tc in self.testcases():
            self.set_testcase_prio(0, tc['Id'] - 1)

        self.scheduled_testcases = []


class IndustrialDatasetScenarioProvider():
    """
    Scenario provider to process CSV files for experimental evaluation of RETECS.

    Required columns are `self.tc_fieldnames` plus ['Verdict', 'Cycle']
    """

    def __init__(self, tcfile, sched_time_ratio=0.5):

        super(IndustrialDatasetScenarioProvider, self).__init__()

        # os.path.basename 是返回最后的文件名
        # os. path.splitext() 将文件名与后缀分割
        # self.basename 最后是文件名字不带后缀，形如 infrol
        self.basename = os.path.splitext(os.path.basename(tcfile))[0]
        self.name = self.basename

        # 尝试将['LastRun']列作为单独的日期列进行分析。
        self.tcdf = pd.read_csv(tcfile, error_bad_lines=False, sep=';', parse_dates=['LastRun'])

        # json.loads() 可以识别出字符串中的json格式：去掉引号，并变成通用的json，所有语言都识别
        self.tcdf['LastResults'] = self.tcdf['LastResults'].apply(json.loads)
        #self.tcdf =self.tcdf.loc[self.tcdf.Cycle <= 3]
        #print(self.tcdf)
        # {测试执行：执行结果} 字典对
        self.solutions = dict(zip(self.tcdf['Id'].tolist(), self.tcdf['Verdict'].tolist()))

        self.cycle = 0

        # Timestamp('2015-02-13 16:13:00') 最早执行时间
        self.maxtime = min(self.tcdf.LastRun)

        # 最大循环次数 inforl数据集是320
        self.max_cycles = max(self.tcdf.Cycle)

        # 场景，环境
        self.scenario = None

        # 执行时间比例 0.5
        self.avail_time_ratio = sched_time_ratio

        self.tc_fieldnames = ['Id', 'Name', 'Duration', 'CalcPrio', 'LastRun', 'LastResults', 'Verdict']

    # name_suffix : 名字 后缀
    def get(self, name_suffix=None):
        self.cycle += 1

        # 当 enumerate该类时，会调用get函数
        # print("执行get函数，cycle=",self.cycle)

        # 超过最大循环，返回return
        if self.cycle > self.max_cycles:
            self.scenario = None
            return None

        # 按标签或布尔数组访问一组行和列。
        # 获取 tcdf.Cycle 等于当前cycle的数据
        cycledf = self.tcdf.loc[self.tcdf.Cycle == self.cycle]

        # 将每行（键为 tc_fieldnames）转为字典
        seltc = cycledf[self.tc_fieldnames].to_dict(orient='record')

        if name_suffix is None:
            # timedelta(days=1) : 1 day, 0:00:00
            # 相当于日期延后一天
            # 此方法的返回类型是日期的ISO 8601格式的字符串。 即1/1/1使用0001/01/01表示
            name_suffix = (self.maxtime + timedelta(days=1)).isoformat()

        # 计算该循环内，所有测试用例执行的总时间 和 相应比例时间
        req_time = sum([tc['Duration'] for tc in seltc])
        total_time = req_time * self.avail_time_ratio

        # 当前循环的 {id：verdict} 字典组合
        selsol = dict(zip(cycledf['Id'].tolist(), cycledf['Verdict'].tolist()))

        self.scenario = VirtualScenario(testcases=seltc, solutions=selsol, name_suffix=name_suffix,
                                        available_time=total_time, schedule_date=self.maxtime + timedelta(days=1))

        #  maxtime设置为当前循环最后一次执行时间
        self.maxtime = seltc[-1]['LastRun']

        return self.scenario

    def get_validation(self):
        """ Validation data sets are not supported for this provider """
        return []

        # Generator functions

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        sc = self.get()

        if sc is None:
            raise StopIteration()

        return sc


def get_scenario(name):
    # if name == 'incremental':
    #     sc = IncrementalScenarioProvider(episode_length=CI_CYCLES)
    if name == 'paintcontrol':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/paintcontrol.csv')
    elif name == 'iofrol':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/iofrol.csv')
    elif name == 'gsdtsr':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/gsdtsr.csv')
    elif name == 'dspace':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/dspace.csv')
    elif name == 'google_auto':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/google_auto.csv')
    elif name == 'apache_drill':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/apache_drill.csv')
    elif name == 'apache_parquet':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/apache_parquet.csv')
    elif name == 'apache_commons':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/apache_commons.csv')
    elif name == 'apache_hive':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/apache_hive.csv')
    elif name == 'apache_tajo':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/apache_tajo.csv')
    elif name == 'google_closure':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/google_closure.csv')
    elif name == 'google_guava':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/google_guava.csv')
    elif name == 'mybatis':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/mybatis.csv')
    elif name == 'rails':
        sc = IndustrialDatasetScenarioProvider(tcfile='../dataset/rails.csv')

    return sc


