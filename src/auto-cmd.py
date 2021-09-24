# python version: 3.8.5
#
# test env: wsl2 ubuntu
#
# TODO: change path and commamd

from json.decoder import JSONDecodeError
import subprocess
import random
import json

# Need to change
lane_path = "/home/ubuntu/qry-codespace/city/data/lanes-id.json"
access_path = "/home/ubuntu/qry-codespace/city/data/access.txt"
output_path = "/home/ubuntu/qry-codespace/city/data/output"
save_result_path = "/home/ubuntu/qry-codespace/city/data/save-result.json"

# cmd
my_command = "sudo docker run --mount type=bind,source=" + output_path + ",target=/output --mount type=bind,source=" + access_path + ",target=/access.txt --rm git.tsingroc.com:5050/release/cup2109:latest"

class AutoCmd:

    waiting_time: int

    lanes_length: int

    shut_down_flag: bool

    def __init__(self, waiting_time: int, lanes_length: int) -> None:
        self.waiting_time = waiting_time
        self.lanes_length = lanes_length
        self.shut_down_flag = False

    def generate_access(self, lane_path: str, access_path: str) -> list:
        my_access = []
        try:
            with open(lane_path, "r") as f:
                lanes: list = json.load(f)
                my_access = random.sample(lanes, random.randint(1, 10))
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

    def runcmd(self, command: str, output_path: str, save_result_path: str, list_: list):
        print("running: max waiting time =", self.waiting_time)
        ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=self.waiting_time)    
        if ret.returncode == 0:
            print("success:", end=" ")
            self.get_result(output_path, save_result_path, list_)
        else:
            print("error:",ret)
            self.waiting_time += 20
            if self.waiting_time >= 600:
                self.shut_down_flag = True
            

    def get_result(self, output_path: str, save_result_path: str, access_list: list):
        result_list = []
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
        result_list.append([first_line, access_list])
        with open(save_result_path, "w") as f:
            json.dump(result_list, f)
            f.close()

if __name__ == "__main__":
    my_auto_cmd = AutoCmd(waiting_time=420, lanes_length=35189)
    cycle_times = 10
    for i in range(cycle_times):
        list_ = my_auto_cmd.generate_access(lane_path ,access_path)
        my_auto_cmd.runcmd(my_command, output_path, save_result_path, list_)
        if (my_auto_cmd.shut_down_flag):
            print("TimeError: break in " + str(i + 1) + "/" + cycle_times)
            break
        print("process: " + str(i + 1) + "/" + cycle_times)

# cmd: nohup python3 /home/ubuntu/qry-codespace/city/src/auto-cmd.py &