# python version:3.6.9
#
# test env: ubuntu 18.04
# 
# TODO: looking for the busiest lane
import json
import random
import subprocess as sp
import numpy as np
import os

traj_path = "../data/output/traj.txt"
access_path = "../data/access.txt"
accesiable_path = "../data/accesiable.json"
save_path = "../data/zsy-result1.json"
lanes_path = "../data/lanes-id.json"
time_path = "../data/output/time.txt"
cmd = "sudo docker run --mount type=bind,source=/home/ubuntu/zsy-space/data/output,target=/output \
    --mount type=bind,source=/home/ubuntu/zsy-space/data/access.txt,target=/access.txt --rm \
    git.tsingroc.com:5050/release/cup2109:latest"

step = 40

class Busiest:

    traj_path: str

    access_path: str

    def __init__(self, traj_path: str, access_path: str) -> None:
        self.traj_path = traj_path
        self.access_path = access_path

    def get_busylane(self, step: int) -> list:                          
        lane_dict = {}                                                  # step表示获得的车流量较大车道的数量
        with open(traj_path, 'r') as f:
            line = f.readline()
            for i in range(719):                                        # 719是traj.txt文件中要读取的行数
                line = f.readline()
                data_list = line.split(' ')
                data_list = data_list[1:]
                index = 0
                while(index < len(data_list) - 1):              
                    car_id = data_list[index]                           # 统计每条车道上走过的车的id
                    lane_id = data_list[index + 1]
                    if(lane_dict.__contains__(lane_id) == False):       # 判断该车辆id是否统计过
                        lane_dict[lane_id] = []
                        lane_dict[lane_id].append(car_id)
                    elif(car_id not in lane_dict[lane_id]):             # 若没统计过，则将车辆id加入到对应的队列中
                        lane_dict[lane_id].append(car_id)
                    index += 3
            f.close()
        for key in lane_dict:
            lane_dict[key] = len(lane_dict[key])
        my_dict = sorted(lane_dict.items(), key = lambda x:x[1], reverse=True)  # 按每条车道统计的车辆id个数排序
        with open("./rank.json", 'w') as f:
            json.dump(my_dict,f)
            f.close()
        my_dict = my_dict[100: 100 + step]                              # 100这个数可修改为合适值
        my_keys = []
        for i in range(step):                                           # 构建选中的车道id列表
            t = my_dict[i]
            my_keys.append(int(t[0]))
        return my_keys

if __name__ == "__main__":
    my_busiest = Busiest(traj_path, access_path)
    # print(my_busiest.get_busylane(20))
    with open(accesiable_path, 'r') as f:
        access_list = json.load(f)
        f.close()
    times = 15
    access = []                                                         # 初始化要限流的车道id列表为空
    result_dict = {}                                                    # 保存参数和结果的字典，用于打印信息
    os.remove(traj_path)
    os.remove(time_path)
    ret = sp.run(cmd,shell=True,stdout=sp.PIPE,stderr=sp.PIPE,encoding="utf-8",timeout=480)
    for i in range(times):
        print("开始运行第"+str(i+1)+"轮")
        busiest_lanes = my_busiest.get_busylane(8)                              # 获取车流量前n～n+8的车道
        print(busiest_lanes)
        i = 0
        while(i < len(busiest_lanes)):
            if(busiest_lanes[i] not in access_list):
                busiest_lanes.remove(busiest_lanes[i])                          # 去除不能限流的车道
            else:
                i += 1
        print(busiest_lanes)
        access = access + busiest_lanes                                         # 增加限流车道
        with open(access_path ,'w') as f:
            for i in range(len(access)):
                f.write(str(access[i])+'\n')
            f.close()
        os.remove(traj_path)
        os.remove(time_path)
        ret = sp.run(cmd,shell=True,stdout=sp.PIPE,stderr=sp.PIPE,encoding="utf-8",timeout=480)
        if(ret.returncode == 0):
            print("success!")
        else:
            print("error:",ret)
        with open(time_path, 'r') as f:
            result = f.readline().splitlines()
            f.close()
        print(result)
        result_dict[str(result)] = access
    with open(save_path,'w') as f:
        json.dump(result_dict, f)
        f.close()