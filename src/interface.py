import json
import numpy as np

f = open("map.json", encoding="utf-8")
map_list = json.load(f)
f.close()

# 获取json所有类型数量
map_dict = {}
for item in map_list:
    if item['class'] in map_dict:
        map_dict[item['class']] += 1
    else:
        map_dict[item['class']] = 1

print(map_dict)

# class_list = [item['class'] for item in map_list]
# with open("class.json", "w") as f:
#     f.write(json.dumps(class_list))
#     f.close()

lane_list = [item['data']['length'] for item in map_list[map_dict['header']: map_dict['lane'] + 1]]
print("avg len = ", np.mean(lane_list))
