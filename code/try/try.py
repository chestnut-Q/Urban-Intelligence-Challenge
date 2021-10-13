import json
import numpy
from json.decoder import JSONDecodeError
import subprocess
import random
import time
import math
import os

lane_path = "lanes-id.json"
access_path = os.sep.join([os.getcwd(), 'access.txt'])
output_path = os.sep.join([os.getcwd(), 'output'])
save_result_path = "save-result.json"
save_best_path = "save_best"

# cmd，如果报错可能不需要加sudo
my_command = "sudo docker run --mount type=bind,source=" + output_path + ",target=/output --mount type=bind,source=" + access_path + ",target=/access.txt --rm git.tsingroc.com:5050/release/cup2109:latest"

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
                # 设置增车道和减车道的阈值
                if len(start_state[1]) < 50:
                    threshold = 1
                    # threshold = 0
                elif len(start_state[1]) < 200:
                    threshold = 0.7
                elif len(start_state[1]) < 350:
                    threshold = 0.5
                else:
                    threshold = 0.3
                if p > threshold: # 随机减少车道
                    my_access = random.sample(start_state[1], random.randint(int(len(start_state[1])*19/20), int(len(start_state[1]))))
                    print('Too many lanes || Reduce lane')
                else: # 随机增加车道
                    # my_access = random.sample(start_state[1], random.randint(int(len(start_state[1])/2), len(start_state[1])))
                    randomroad = random.sample(lanes, random.randint(1, 30)) 
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
        with open(access_path, "w") as f: # 将限行方案写入access.txt
            list_ = []
            for item in my_access:
                f.write(str(item) + '\n')
                list_.append(item)
            f.close()
        return list_

    def runcmd(self, command: str, output_path: str, save_result_path: str, list_: list, start_state, i): # 仿真
        print("running: max waiting time =", self.waiting_time)
        ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=self.waiting_time)    
        if ret.returncode == 0:
            print("success:", end=" ")
            self.get_result(output_path, os.sep.join([os.getcwd(), 'result', 'save-result' + str(i) + '.json']), list_, start_state)
        else:
            print("error:",ret)
            self.waiting_time += 20
            if self.waiting_time >= 600:
                self.shut_down_flag = True
            

    def get_result(self, output_path: str, save_result_path: str, access_list: list, start_state): # 读取仿真结果并写入文件
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
    if not os.path.exists(os.sep.join([os.getcwd(), 'result'])):
        os.makedirs(os.sep.join([os.getcwd(), 'result']))
    if not os.path.exists(os.sep.join([os.getcwd(), 'save_best'])):
        os.makedirs(os.sep.join([os.getcwd(), 'save_best']))
    my_auto_cmd = AutoCmd(waiting_time=420, lanes_length=35189)
    cycle_times = 40 # 遗传轮数
    value = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 初始化每个种子状态遗传的状态数
    with open('start.json', 'r') as f:
        state_list = json.load(f)
    for i in range(cycle_times):
        print("process: " + str(i) + "/" + str(cycle_times))
        for statenum in range(len(state_list)):
            state = state_list[statenum]
            print('==============')
            print('state: ' + str(statenum) + '/' + str(len(state_list)))
            print('Previous speed: ' + str(state[0]))
            # print(state[1])
            for k in range(value[statenum]):
                starttime = time.time()
                print("try: " + str(k) + "/" + str(value[statenum]))
                list_ = my_auto_cmd.generate_access(lane_path ,access_path, state) # 将限行车道写入access.txt
                # break
                my_auto_cmd.runcmd(my_command, output_path, save_result_path, list_, state[0], i) # 仿真测试结果，保存结果
                if (my_auto_cmd.shut_down_flag):
                    print("TimeError: break in " + str(i + 1) + "/" + str(cycle_times))
                    break
                print('Time: ' + str(time.time() - starttime) + 's')
                # print("process: " + str(i + 1) + "/" + str(cycle_times))
            # break
        with open(os.sep.join([os.getcwd(), 'result', 'save-result' + str(i) + '.json']), 'r') as f:
            new_state = json.load(f)
            for state in new_state:
                if (float(state[0]) <= state_list[-1][0] + 6):
                    if state not in state_list:
                        state_list.append([float(state[0]), state[1]])
            rlist = sorted(state_list, key=(lambda x: [x[0]])) # 寻找前20个最快的限行方案
            n = min(20, len(rlist))
            state_list = rlist[0:n]
            state_list = sorted(state_list, key = (lambda x: len(x[1])))
            state_list = state_list[0:min(10, len(state_list))] # 取前10个限行车道最少的限行方案
            grade = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 奖励与惩罚因子
            for state in state_list:
                if len(state) == 3:
                    last_state = state[2]
                    for k in len(state_list):
                        if state_list[k][0] == last_state: # 如果该状态上一轮变异出较好的结果，给予奖励
                            grade[k] = grade[k] + 2
            for k in range(10):
                value[k] = 3 + grade[k] * 2 # 更新下一轮遗传的状态数
        print("=========================")
        print("process: " + str(i) + "  best time " + str(state_list[0][0]))
        print("=========================")
        with open(os.sep.join(["save_best", "best_result" + str(i) + ".json"]), 'w') as f: # 保存本轮最好结果
            json.dump(state_list, f)
            f.close()
# cmd: nohup python3 /home/ubuntu/qry-codespace/city/src/auto-cmd.py &