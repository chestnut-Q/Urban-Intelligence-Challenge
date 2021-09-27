import json
import numpy as np
import os

infolder = r"..\data"
outfolder = r"..\data"
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

resultlist = []
filelist = ['save-result.json', 'save-result0.json', 'save-result2.json', 'save-result5.json', 'start.json']
for f in filelist:
    print(os.sep.join([infolder, f]))
    with open(os.sep.join([infolder, f]),'r') as fi:
        load_dict = json.load(fi)
        l = len(load_dict)
        for k in range(l):
            ndict = load_dict[k]
            if float(ndict[0]) > 5000:
                continue
            resultlist.append([str(ndict[0]), ndict[1]])

# print(resultlist[0])
print(type(resultlist))
print(resultlist[1])
# test = list(set(resultlist))
test = []
for result in resultlist:
    if result not in test:
        test.append(result)

# val = resultlist[int(len(resultlist)*7/8):-1]

rlist = sorted(test, key=(lambda x: [float(x[0])]))
# print(rlist[0])
# print(rlist[1])
# print(rlist[2])
# print(rlist[3])
# print(rlist[-1])

outfile = os.sep.join([outfolder, 'allresult.json'])
print('result num: ' + str(len(rlist)))
# # if not os.path.exists(outfile):

# label = []
# for p in range(int(len(rlist)*1/2)):
#     q = p + int(len(rlist)*1/2)
#     while q < len(rlist):
#         dic1 = {'label':0, 'cond1':rlist[p][1], 'speed1':rlist[p][0], 'cond2':rlist[q][1], 'speed2':rlist[q][0]}
#         dic2 = {'label':1, 'cond1':rlist[q][1], 'speed1':rlist[q][0], 'cond2':rlist[p][1], 'speed2':rlist[p][0]}
#         label.append(dic1)
#         label.append(dic2)
#         q = q + 1


rstart = rlist
with open(outfile, "w") as f:
    # print('true')
    json.dump(rstart, f)
    f.close()

with open(os.sep.join([outfolder, 'readable_result.json']), 'w') as f:
    for result in rstart:
        json.dump(result, f)
        f.write('\n')
    f.close()


# rlist = sorted(val, key=(lambda x: [x[0]]))
# print(val)
# # print(rlist[0])
# # print(rlist[1])

# outfile = os.sep.join([outfolder, 'val' + str(len(os.listdir(outfolder))) + '.json'])
# print(outfile)
# # if not os.path.exists(outfile):

# label = []
# for p in range(int(len(rlist)*1/3)):
#     q = p + int(len(rlist)*2/3)
#     while q < len(rlist):
#         dic1 = {'label':0, 'cond1':rlist[p][1], 'cond2':rlist[q][1]}
#         dic2 = {'label':1, 'cond1':rlist[q][1], 'cond2':rlist[p][1]}
#         label.append(dic1)
#         label.append(dic2)
#         q = q + 1
        

# with open(outfile, "w") as f:
#     print('true')
#     json.dump(label, f)
#     f.close()
