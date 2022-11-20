
# -*- coding: UTF-8 -*-
"""
@author:cfl
@file:count_file.py
@time:2022/03/13
@software:PyCharm
"""

import os

def main():
    print("results文件夹下共有文件数：",len(os.listdir("/home/chenfanliang/myretecs/novlty/results")))


if __name__ == '__main__':
    main()
