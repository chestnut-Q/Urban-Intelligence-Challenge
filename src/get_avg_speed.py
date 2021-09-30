# 1. 函数参数: traj_path = traj.txt路径, spped_path = 保存结果路径(文件可以没有)
#
# 2. json数据格式: (车道id, [平均速度, 车流量]) 按平均速度降序排列
# 
# 3. 预计用时5min

import json
import time

traj_path = "/home/ubuntu/city-competition/output/traj.txt"
speed_path = "/home/ubuntu/qry-codespace/city/data/speed.json"

def avg_speed(traj_path: str, speed_path: str) -> None:

    print("running: expected waiting time = 300")
    time_start=time.time()

    with open(traj_path, "r", encoding="utf-8") as f:
        list_ = f.readlines()
        f.close()

    cars_num: int = 77248
    lanes_num: int = 35189
    lanes_info = [[0, 0, 0] for i in range(lanes_num)]  # [0]保存总距离 [1]保存总时间 [2]保存车流量
    last_car_info = [[-1, []] for i in range(cars_num)]  # [0]保存车道 [1]保存距离
    my_car_info = [[-1, []] for i in range(cars_num)]
    stop_cars_list = []

    for time_index in range(1, len(list_)):
        my_list: list = list_[time_index].replace('\n', '').split(' ')
        # print("length:", len(my_list))
        for car_index in range(1, len(my_list), 3):
            car_id = int(my_list[car_index])
            if car_id in stop_cars_list:
                continue
            my_car_info[car_id][0] = int(my_list[car_index + 1])
            my_car_info[car_id][1] = float(my_list[car_index + 2])

            def update_lanes_info():
                lane_id = last_car_info[car_id][0]
                if len(last_car_info[car_id][1]) > 1:
                    lanes_info[lane_id][0] += abs(float(last_car_info[car_id][1][-1]) - float(last_car_info[car_id][1][0]))
                    lanes_info[lane_id][1] += (len(last_car_info[car_id][1]) - 1) * 10
                    lanes_info[lane_id][2] += 1
                last_car_info[car_id][0] = -1
                last_car_info[car_id][1].clear()

            if last_car_info[car_id][0] == int(my_list[car_index + 1]):
                # 该车连续在该车道上行驶
                if len(last_car_info[car_id][1]) > 0 and last_car_info[car_id][1][-1] == my_list[car_index + 2]:
                    # 该车停止
                    update_lanes_info()
                    stop_cars_list.append(car_id)
                else:
                    last_car_info[car_id][1].append(my_list[car_index + 2])
            else:
                # 换车道，结算平均速度
                update_lanes_info()
                last_car_info[car_id][0] = int(my_list[car_index + 1])
                last_car_info[car_id][1].append(my_list[car_index + 2])

    lanes_avg_speed = [[x[1] / x[0] if x[0] != 0 else 0, x[2]] for x in lanes_info]
    lanes_avg_speed_sort = sorted(enumerate(lanes_avg_speed), key=lambda x: x[1], reverse=True)
    # for item in lanes_avg_speed_sort[:10]:
    #     print(item)

    with open(speed_path, "w", encoding="utf-8") as f:
        json.dump(lanes_avg_speed_sort, f)
        f.close()

    print("done: time =", time.time() - time_start)

if __name__ == "__main__":
    avg_speed(traj_path, speed_path)

# cmd: nohup python3 -u /home/ubuntu/qry-codespace/city/src/get_avg_speed.py > /home/ubuntu/qry-codespace/city/store/output.log &