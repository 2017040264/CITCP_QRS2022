
# -*- coding: UTF-8 -*-
"""
@author:cfl
@file:myvisualization.py
@time:2022/03/10
@software:PyCharm
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob
import seaborn as sns


USE_LATEX=False
DATA_DIR = 'results'
FIGURE_DIR = 'results_csv'

def load_stats_dataframe(files, aggregated_results=None):
    # print(os.path.exists(aggregated_results))
    # print(os.path.getmtime(aggregated_results)) # 文件的创建时间
    # aggregated 是压缩文件，由该函数最后生成
    if os.path.exists(aggregated_results) and all(
            [os.path.getmtime(f) < os.path.getmtime(aggregated_results) for f in files]):
        return pd.read_pickle(aggregated_results)

    df = pd.DataFrame()

    for f in files:
        tmp_dict = pd.read_pickle(f)

        tmp_dict['iteration'] = f.split('_')[-2]

        del tmp_dict['result']

        tmp_df = pd.DataFrame.from_dict(tmp_dict)
        # print(len(tmp_df))
        df = pd.concat([df, tmp_df])

    if aggregated_results:
        # to_pickle：文件压缩
        df.to_pickle(aggregated_results)

    return df


# reward_names =  {
#         "APHF":"APHF",
#         "APHF_TW5":"APHF_TW5",
#
#         "APHF_NV":"APHF_NV",
#         "APHF_NV_MIN": "APHF_NV_MIN",
#         "APHF_NV_AVE": "APHF_NV_AVE",
#
#         "APHF_NV50": "APHF_NV50",
#         "APHF_NV50_MIN": "APHF_NV50_MIN",
#         "APHF_NV50_AVE": "APHF_NV50_AVE",
#
#         "APHF_NVH50": "APHF_NVH50",
#         "APHF_NVH50_MIN": "APHF_NVH50_MIN",
#         "APHF_NVH50_AVE": "APHF_NVH50_AVE",
#
#         "APHF_NV25": "APHF_NV25",
#         "APHF_NV25_MIN": "APHF_NV25_MIN",
#         "APHF_NV25_AVE": "APHF_NV25_AVE",
#
#         "APHF_NV025": "APHF_NV025",
#         "APHF_NV025_MIN": "APHF_NV025_MIN",
#         "APHF_NV025_AVE": "APHF_NV025_AVE",
#
#         "APHF_NVH25": "APHF_NVH25",
#         "APHF_NVH25_MIN": "APHF_NVH25_MIN",
#         "APHF_NVH25_AVE": "APHF_NVH25_AVE",
#
#         "APHF_NV050": "APHF_NV050",
#         "APHF_NV050_MIN": "APHF_NV050_MIN",
#         "APHF_NV050_AVE": "APHF_NV050_AVE",
#     }
#
# env_names = {
#                 'iofrol': 'ABB IOF/ROL',
#                 "paintcontrol":"PaintControl",
#                 'gsdtsr': 'GSDTSR',
#                 "apache_drill": 'Apache_Drill',
#                 "google_auto": "Google_Auto",
#                 "dspace": "Dspace",
#                 "apache_parquet": "Apache_Parquet",
#                 'apache_commons': 'Apache_Commons',
#                 'apache_hive':'Apache_Hive',
#                 'apache_tajo':'Apache_Tajo',
#                 'google_closure':'Google_Closure',
#                 'google_guava':'Google_Guava',
#                 'mybatis':'Mybatis',
#                 'rails':'Rails'
# }


def visualize(p):
    search_pattern = 'rq_*_stats.p'
    filename = p+'_rq'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    # print('iteration_results = ',iteration_results)
    aggregated_results = os.path.join(FIGURE_DIR, filename)
    # print('aggregated_results = ',aggregated_results)

    df = load_stats_dataframe(iteration_results, aggregated_results)
    # print('df = ',df)

    pure_df = df[(df['detected'] + df['missed'] > 0)]
    mmm_df = pure_df.groupby(['env', 'rewardfun' , "agent"], as_index=False).mean()
    mmm_df.to_csv(os.path.join(FIGURE_DIR, p+'_result.csv'))




