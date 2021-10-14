# python version:3.6.9
#
# test env: ubuntu 18.04
# 
# TODO:

from json.decoder import JSONDecodeError
import json
import time

mapPath = "../data/map.json"

if __name__ == "__main__":
    print("start!")
    start = time.clock()
    with open(mapPath,'r') as f:
        mapData = json.load(f)
        f.close()
    laneNum = 0
    laneList = []
    for i in range(len(mapData)):                   # 统计车道的数量
        temp = mapData[i]
        if(temp['class']=='lane'):
            tempDict = temp['data']
            laneList.append(int(tempDict['id']))
            laneNum += 1
    print(laneNum)
    print(laneList[0:3])
    lane2poi = []
    for i in range(len(mapData)):                   # 统计和兴趣点相连的车道
        temp = mapData[i]
        if(temp['class']=='poi'):
            tempData = temp['data']
            tDict = tempData['driving_position']
            laneId = int(tDict["lane_id"])
            if(laneId not in lane2poi):             # 加入到列表中
                lane2poi.append(laneId)
    print(len(lane2poi))
    access_lane = list(set(laneList) - set(lane2poi))   # 从车道列表中删掉和兴趣点相连的车道
    print(len(access_lane))
    end = time.clock()
    print("Time Consume: %ss"%(end-start))