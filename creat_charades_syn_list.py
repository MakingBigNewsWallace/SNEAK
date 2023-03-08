import os
import numpy as np
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import tqdm

f=open("D:\LAB\local_adv\VSLNet\data\Charades/charades_sta_test.txt","r")
data=f.readlines()
f.close()
data_len=len(data)
syn_num=5
# print(data[vid_list[0]]["timestamps"][sample_index])
syn_list=[]
syn_list_with_ori={}
# print(vid_list)
total_len=0
aug = naw.SynonymAug(aug_src='wordnet')
with open("D:\LAB\local_adv\VSLNet\data\Charades/test_syn.txt","w") as fslo:
    for i in range(data_len):
        cur_list=[]
        vid_info=data[i].split("##")[0]
        sentence=data[i].split("##")[1][:-1]
        augmented_text = aug.augment(sentence, n=syn_num)
        for j in range(syn_num):
            cur_list.append(vid_info+"##"+augmented_text[j]+"\n")
            fslo.write(vid_info+"##"+augmented_text[j]+"\n")
        syn_list.extend(cur_list)
        print(i)
    # syn_list[vid_list[i]]={}
    # syn_list_with_ori[vid_list[i]] = {}
    # cur_len=len(data[vid_list[i]]["timestamps"])#get sample count
    # total_len=total_len+cur_len
    # syn_list[vid_list[i]]["timestamps"]=[pair for pair in data[vid_list[i]]["timestamps"] for count in range(syn_num)]
    # # syn_list_with_ori[vid_list[i]]["timestamps"] = [pair for pair in data[vid_list[i]]["timestamps"] for count in range(syn_num+1)]

    # cur_syn=[]
    # cur_syn_with_ori=[]
    # # aug = naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-s-all')
    # for j in range(cur_len):
    #     ori_sent=data[vid_list[i]]["sentences"][j]
    #     augmented_text = aug.augment(data[vid_list[i]]["sentences"][j], n=syn_num)
    #     # cur_syn_with_ori.extend([ori_sent])
    #     # cur_syn_with_ori.extend(augmented_text)
    #     cur_syn.extend(augmented_text)
    # syn_list[vid_list[i]]["sentences"]=cur_syn
    # # syn_list_with_ori[vid_list[i]]["sentences"] = cur_syn_with_ori
    #
    # syn_list[vid_list[i]]["fps"] = 29.4
    # # syn_list_with_ori[vid_list[i]]["fps"] = 29.4
    # syn_list[vid_list[i]]["num_frames"] = data[vid_list[i]]["num_frames"]
    # # syn_list_with_ori[vid_list[i]]["num_frames"] = data[vid_list[i]]["num_frames"]
print(len(syn_list))
# with open("D:\LAB\local_adv\VSLNet\data\Charades/train_syn.txt","w+") as fslo:
#     fslo.write(str(syn_list))