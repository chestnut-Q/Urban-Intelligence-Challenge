import json
import numpy
from json.decoder import JSONDecodeError
import subprocess
import random
import time
import math
import os

# 将以下路径修改成本地的路径
lane_path = "lanes-id.json" # lane-id文件
access_path = r"D:\学习\2021秋\城市认知智能挑战赛\access.txt" # access.txt路径
output_path = r"D:\学习\2021秋\城市认知智能挑战赛\输出结果" # output文件夹的路径

# TODO: 存储每一次最优的结果；打印每一轮最小的时间；去掉减少车道时的输出；当一个初始状态的后继状态都比他好时，破例将其保留
# TODO: 测试减少初始状态； 减少车道时，至少保留原有的2/3
# cmd
my_command = "docker run --mount type=bind,source=" + output_path + ",target=/output --mount type=bind,source=" + access_path + ",target=/access.txt --rm git.tsingroc.com:5050/release/cup2109:latest"

class AutoCmd:

    waiting_time: int

    lanes_length: int

    shut_down_flag: bool

    def __init__(self, waiting_time: int, lanes_length: int) -> None:
        self.waiting_time = waiting_time
        self.lanes_length = lanes_length
        self.shut_down_flag = False

    def generate_access(self, lane_path: str, access_path: str, start_state) -> list:
        my_access = []
        try:
            with open(lane_path, "r") as f:
                lanes: list = json.load(f)
                p =  random.random()
                if len(start_state[1]) < 50:
                    threshold = 0.9
                    # threshold = 0
                elif len(start_state[1]) < 250:
                    threshold = 0.7
                elif len(start_state[1]) < 300:
                    threshold = 0.5
                else:
                    threshold = 0.3
                if p > threshold:
                    my_access = random.sample(start_state[1], random.randint(int(len(start_state[1])*2/3), len(start_state[1])))
                    print('Too many lanes || Reduce lane')
                else:
                    # my_access = random.sample(start_state[1], random.randint(int(len(start_state[1])/2), len(start_state[1])))
                    randomroad = random.sample(lanes, random.randint(10, 30))
                    last_access = start_state[1]
                    my_access = list(set(last_access + randomroad))
                f.close()
        except JSONDecodeError or FileNotFoundError:
            lanes_list = []
            for i in range(self.lanes_length):
                lanes_list.append(i)
            with open(lane_path, "w") as f:
                json.dump(lanes_list, f)
                f.close()
        with open(access_path, "w") as f:
            list_ = []
            for item in my_access:
                f.write(str(item) + '\n')
                list_.append(item)
            f.close()
        return list_

    def runcmd(self, command: str, output_path: str, save_result_path: str, list_: list, start_state, i):
        print("running: max waiting time =", self.waiting_time)
        ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=self.waiting_time)    
        if ret.returncode == 0:
            print("success:", end=" ")
            self.get_result(output_path, 'save-result' + str(i) + '.json', list_, start_state)
        else:
            print("error:",ret)
            self.waiting_time += 20
            if self.waiting_time >= 600:
                self.shut_down_flag = True
            

    def get_result(self, output_path: str, save_result_path: str, access_list: list, start_state):
        result_list = []
        if os.path.exists(save_result_path):
            try:
                with open(save_result_path, "r", encoding="utf-8") as f:
                    result_list: list = json.load(f)
                    f.close()        
            except JSONDecodeError or FileNotFoundError:
                with open(save_result_path, "w") as f:
                    f.write("[]")
                    f.close()

        time_path = output_path + "/time.txt"
        with open(time_path, "r") as f:
            first_line = f.readline().strip()
            print("avg time =", first_line)
            f.close()
        result_list.append([first_line, access_list, start_state])
        with open(save_result_path, "w") as f:
            json.dump(result_list, f)
            f.close()

if __name__ == "__main__":
    my_auto_cmd = AutoCmd(waiting_time=420, lanes_length=35189)
    cycle_times = 4
    value = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    with open('start.json', 'r') as f:
        state_list = json.load(f)
    for i in range(cycle_times):
        print("process: " + str(i) + "/" + str(cycle_times))
        for statenum in range(len(state_list)):
            state = state_list[statenum]
            print('Previous speed: ' + str(state[0]))
            # print(state[1])
            for k in range(value[statenum]):
                starttime = time.time()
                print("try: " + str(k) + "/" + str(value[statenum]))
                list_ = my_auto_cmd.generate_access(lane_path ,access_path, state)
                # break
                my_auto_cmd.runcmd(my_command, output_path, save_result_path, list_, state[0], i)
                if (my_auto_cmd.shut_down_flag):
                    print("TimeError: break in " + str(i + 1) + "/" + str(cycle_times))
                    break
                print('Time: ' + str(time.time() - starttime) + 's')
                # print("process: " + str(i + 1) + "/" + str(cycle_times))
            # break
        with open('save-result' + str(i) + '.json', 'r') as f:
            new_state = json.load(f)
            for state in new_state:
                if (float(state[0]) <= state_list[-1][0] + 6):
                    state_list.append([float(state[0]), state[1]])
            rlist = sorted(state_list, key=(lambda x: [x[0]]))
            n = min(10, len(rlist))
            state_list = rlist[0:n]
            state_list = sorted(state_list, key = (lambda x: len(x[1])))
            state_list = state_list[0:min(10, len(state_list))]
            grade = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for state in state_list:
                if len(state) == 3:
                    last_state = state[2]
                    for k in len(state_list):
                        if state_list[k][0] == last_state:
                            grade[k] = grade[k] + 1
            for k in range(10):
                value[k] = 3 + grade[k] * 2
        print("=========================")
        print("process: " + str(i) + "  best time " + str(state_list[0][0]))
        print("=========================")
        bestpath = os.sep.join([os.getcwd(), 'best_result' + str(i) + '.json'])
        with open(bestpath, 'w') as f:
            json.dump(state_list, f)
            f.close()
# cmd: nohup python3 /home/ubuntu/qry-codespace/city/src/auto-cmd.py &