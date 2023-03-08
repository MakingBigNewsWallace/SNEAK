import os
import numpy as np
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import tqdm

# f=open("D:\LAB\local_adv\VSLNet\data\TACoS/test.json","r")
# data=f.readlines()
# data=data[0]
# data=eval(data)
# vid_list=list(data.keys())
# len_index=[]
# for i in range(len(vid_list)):
#     len_index.append(len(data[vid_list[i]]["timestamps"]))
# print(len_index,len(len_index))
#
# num_list=open("D:\LAB\local_adv\VSLNet\datasets/tacos_new/128/high_acc_num.json","r")
# num=num_list.readlines()
# num=num[0]
# num=eval(num)
# # vid_list=list(data.keys())
# print(len(num))
# record=[]
# for i in range(717):
#     for j in range(0,25):
#         if num[i]<len_index[j]:
#             record.append([j,num[i]])
#             break
#         else: num[i]=num[i]-len_index[j]
#
# print(record)
# recover_list={}
# for i in range(25):#init
#     recover_list[vid_list[i]]={}
#     recover_list[vid_list[i]]["timestamps"] = []
#     recover_list[vid_list[i]]["sentences"]=[]
#     recover_list[vid_list[i]]["fps"]=29.4
#     recover_list[vid_list[i]]["num_frames"]=data[vid_list[i]]["num_frames"]
#
# for i in range(717):
#     recover_list[vid_list[record[i][0]]]["timestamps"].append(data[vid_list[record[i][0]]]["timestamps"][record[i][1]])
#     recover_list[vid_list[record[i][0]]]["sentences"].append(data[vid_list[record[i][0]]]["sentences"][record[i][1]])
#
# high_acc_set=str(recover_list).replace("'", "\"")
# with open("D:\LAB\local_adv\VSLNet\datasets/tacos_new/128/high_acc.json", "w+") as high_acc_list:
#     high_acc_list.write(high_acc_set)

f=open("D:\LAB\local_adv\VSLNet\data\TACoS/test.json","r")
data=f.readlines()
data=data[0]
data=eval(data)
vid_list=list(data.keys())
f.close()
# video_index=0
# sample_index=0#57,17
syn_num=5
# print(data[vid_list[0]]["timestamps"][sample_index])
syn_list={}
syn_list_with_ori={}
# print(vid_list)
total_len=0
for i in range(len(vid_list)):
    syn_list[vid_list[i]]={}
    syn_list_with_ori[vid_list[i]] = {}
    cur_len=len(data[vid_list[i]]["timestamps"])#get sample count
    total_len=total_len+cur_len
    syn_list[vid_list[i]]["timestamps"]=[pair for pair in data[vid_list[i]]["timestamps"] for count in range(syn_num)]
    # # syn_list_with_ori[vid_list[i]]["timestamps"] = [pair for pair in data[vid_list[i]]["timestamps"] for count in range(syn_num+1)]
    aug = naw.SynonymAug(aug_src='wordnet')
    cur_syn=[]
    # cur_syn_with_ori=[]
    # # aug = naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-s-all')
    for j in range(cur_len):
        ori_sent=data[vid_list[i]]["sentences"][j]
        augmented_text = aug.augment(data[vid_list[i]]["sentences"][j], n=syn_num)
    #     # cur_syn_with_ori.extend([ori_sent])
    #     # cur_syn_with_ori.extend(augmented_text)
        cur_syn.extend(augmented_text)
    # cur_str=str([{} for i in range(cur_len)]).format(cur_syn[i])
    syn_list[vid_list[i]]["sentences"]=cur_syn
    # # syn_list_with_ori[vid_list[i]]["sentences"] = cur_syn_with_ori
    #
    syn_list[vid_list[i]]["fps"] = 29.4
    # syn_list_with_ori[vid_list[i]]["fps"] = 29.4
    syn_list[vid_list[i]]["num_frames"] = data[vid_list[i]]["num_frames"]
    # syn_list_with_ori[vid_list[i]]["num_frames"] = data[vid_list[i]]["num_frames"]
print(total_len)

# pick_ori_sample={}
# pick_ori_sample["timestamps"]=[data[vid_list[0]]["timestamps"][sample_index]]*(syn_num)
# #creat syn list for attack
# aug = naw.SynonymAug(aug_src='wordnet')
# # aug = naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-s-all')
# augmented_text = aug.augment(data[vid_list[0]]["sentences"][sample_index],n=syn_num)
# print("Original:")
# print(data[vid_list[0]]["sentences"][sample_index])
# print("Augmented Text:")
# print(augmented_text,"\n")

syn_list=str(syn_list).replace("'","\"")
# syn_list=str(syn_list)
# import json
# syn_list=json.dumps(syn_list)
with open("D:\LAB\local_adv\VSLNet\data\TACoS/test_syn.json","w+") as fsl:
    fsl.write(syn_list)

# syn_list_with_ori=str(syn_list_with_ori).replace("'","\"")
# with open("D:\LAB\local_adv\VSLNet\data\TACoS/high_syn_set_with_ori.json","w+") as fslo:
#     fslo.write(syn_list_with_ori)



# #creat original sample list in compare
# ori_sample={}
# ori_sample["timestamps"]=[data[vid_list[0]]["timestamps"][sample_index]]
# ori_sample["sentences"]=[data[vid_list[0]]["sentences"][sample_index]]
# ori_sample["fps"]=29.4
# ori_sample["num_frames"]=data[vid_list[0]]["num_frames"]
# ori_list={}
# ori_list[vid_list[video_index]]=ori_sample
# ori_list=str(ori_list).replace("'","\"")
#


