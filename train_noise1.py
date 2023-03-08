import os
import cv2
import glob
import json
import torch
import threading
import random
import argparse
import subprocess
import numpy as np
from PIL import Image
from VSLNet.prepare import videotransforms
# from .feature_extractor import InceptionI3d
from VSLNet.prepare.feature_extractor import InceptionI3d
from torchvision import transforms
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

noise_dict = {}
lock = threading.Lock()

strides = 24
nstrides = 16  # max number of strides on single card
top_noise0_count = 32  # select top biggest noise0 strides(n/128)


class Top_trainThread(threading.Thread):
    def __init__(self, GPUID, gpu_count, load_model, top_feats, top_sub_imgs):
        threading.Thread.__init__(self)
        self.GPUID = GPUID
        self.load_model = load_model
        self.gpu_count = gpu_count
        self.top_feats = top_feats
        self.sub_imgs = np.reshape(top_sub_imgs, [-1, 224, 224, 3])  # L,224,224,3

    def run(self):
        print("start Top_training thread：" + str(self.GPUID))
        self.i3d_model = InceptionI3d(400, in_channels=3)
        self.i3d_model.load_state_dict(torch.load(self.load_model))
        self.i3d_model.to(torch.device("cuda:{}".format(self.GPUID)))
        self.i3d_model.train(False)

        noise1 = []
        loss = []
        # for i in range(int(self.sub_feats.shape[0] / nstrides / self.gpu_count * (self.GPUID)),
        #                int(self.sub_feats.shape[0] / nstrides / self.gpu_count * (self.GPUID + 1))):
        for i in range(int(self.top_feats.shape[0] / nstrides)):
            print("start part ", (i + 1), "total {} part".format(int(self.top_feats.shape[0] / nstrides)),
                  "\033[0;32;40m\t--thread On GPU{}\033[0m".format(self.GPUID))
            noise1_1 = np.expand_dims(np.random.normal(0, 1, [strides * nstrides, 224, 224, 3]).transpose([3, 0, 1, 2]),
                                      axis=0).astype(np.float32)
            noise1_1 = torch.from_numpy(noise1_1).cuda(self.GPUID)
            print(noise1_1.device,
                  "\033[0;33;40m\tdevice for thread{}______________________________\033[0m".format(self.GPUID))
            noise1_1 = Variable(noise1_1, requires_grad=True)  # 1,3,16*nstrides,224,224
            sub_imgs_1 = self.sub_imgs[strides * nstrides * (i):strides * nstrides * (i + 1), :, :, :]

            # init image tensor with noise
            img_tensor = torch.from_numpy(np.expand_dims(sub_imgs_1.transpose([3, 0, 1, 2]), axis=0)).cuda(
                self.GPUID)  # (1,3,L,224,224)
            img_tensor_with_noise = (noise1_1 + img_tensor).cuda(self.GPUID)
            # print(img_tensor_with_noise.shape, "img_tensor_with_noise")

            # noise0 = np.squeeze(np.load("/home/gwenbo/local_adv/VSLNet/cur_noise_target.npy"))  # 上一步的noise
            # target_feature = self.top_feats + noise0
            target_feature = self.top_feats[nstrides * i:nstrides * (i + 1), :]  # 得到分段的扰动特征
            target_feature = torch.from_numpy(target_feature).cuda(self.GPUID)
            optimizer = torch.optim.Adam([noise1_1], lr=0.5)
            count = 0
            # max iter 500
            # features = extract_features(img_tensor_with_noise, self.i3d_model, strides,self.GPUID)
            # while np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())) > 4:
            while count <= 2000:
                count += 1
                optimizer.zero_grad()
                features = extract_features(img_tensor_with_noise, self.i3d_model, strides, self.GPUID)
                # LOSS = torch.sum((features - target_feature) ** 2) + noise_lambda * torch.mean(noise1_1 ** 2)
                # LOSS=torch.sum((features-target_feature)**2)+noise_lambda*torch.sum(noise1_1**2)
                LOSS = torch.sum((features - target_feature) ** 2)
                if count == 1:
                    last_loss = LOSS.data.cpu().numpy()
                LOSS.backward()
                optimizer.step()
                # 限制到0-255间
                # noise1_1=torch.clamp(noise1_1,0,255)#这么做会报错，也有问题！
                img_tensor_with_noise = img_tensor + noise1_1
                img_tensor_with_noise = torch.clamp(img_tensor_with_noise, 0, 255)
                if count % 50 == 0:
                    print("\033[0;31;4{}mOn GPU{} iter:{} {} {}\033[0m".format(0 if self.GPUID % 2 == 0 else 7,
                                                                               self.GPUID,
                                                                               count, LOSS.data.cpu().numpy(),
                                                                               last_loss - LOSS.data.cpu().numpy()))
                    last_loss = LOSS.data.cpu().numpy()
            print("\033[0;31;4{}mOn GPU{}\033[0m".format(0 if self.GPUID % 2 == 0 else 7, self.GPUID), count,
                  LOSS.data.cpu().numpy(),
                  np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())))
            if count < 10:
                noise1.append(np.zeros((16 * nstrides, 224, 224, 3)))
                print("add zero only")
            else:
                loss.append(np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())))
                noise1.append(
                    np.round(np.transpose(np.squeeze(noise1_1.data.cpu().numpy()), [1, 2, 3, 0])))  # 使用round转为整数

        loss = np.mean(np.asarray(loss))
        noise1 = np.asarray(noise1)
        print(np.mean(noise1), np.mean(noise1 ** 2), np.sum(noise1 ** 2),
              "Mean /Mean Square/sum square value of noise ", loss,
              "mean loss to target features")
        noise1 = np.reshape(np.array(noise1), [-1, 224, 224, 3])
        lock.acquire()
        noise_dict["{}".format(self.GPUID)] = noise1
        lock.release()
        # return noise1


class Specified_trainThread(threading.Thread):
    def __init__(self, GPUID, load_model, spec_feats, spec_sub_imgs,dataset,iter):
        threading.Thread.__init__(self)
        self.GPUID = GPUID
        self.load_model = load_model
        self.spec_feats = spec_feats
        self.spec_sub_imgs = np.reshape(spec_sub_imgs, [-1, 224, 224, 3])  # L,224,224,3
        if dataset=="Charades":
            # self.i3d_model.replace_logits(157)  # charades has 157 activity types
            self.strides=12
        elif dataset=="TACoS":
            # self.i3d_model.replace_logits(157)  # charades has 157 activity types
            self.strides=16
        self.iter=iter
    def run(self):
        print("start training thread：" + str(self.GPUID))
        self.i3d_model = InceptionI3d(400, in_channels=3)
        self.i3d_model.load_state_dict(torch.load(self.load_model))
        self.i3d_model.to(torch.device("cuda:{}".format(self.GPUID)))
        self.i3d_model.train(False)
        len = self.spec_feats.shape[0]
        noise1 = []
        loss = []

        # for i in range(int(self.top_feats.shape[0]/nstrides)):
        # print("start part ", (i + 1), "total {} part".format(int(self.top_feats.shape[0]/nstrides)),
        #       "\033[0;32;40m\t--thread On GPU{}\033[0m".format(self.GPUID))
        noise1_1 = np.expand_dims(np.random.normal(0, 1, [self.strides * len, 224, 224, 3]).transpose([3, 0, 1, 2]),
                                  axis=0).astype(np.float32)
        noise1_1 = torch.from_numpy(noise1_1).float().cuda(self.GPUID)
        print(noise1_1.device,noise1_1.dtype,
              "\033[0;33;40m\tdevice for thread{}______________________________\033[0m".format(self.GPUID))
        noise1_1 = Variable(noise1_1, requires_grad=True)  # 1,3,16*nstrides,224,224,3

        # init image tensor with noise
        img_tensor = torch.from_numpy(np.expand_dims(self.spec_sub_imgs.transpose([3, 0, 1, 2]), axis=0)).float().cuda(
            self.GPUID)  # (1,3,L,224,224)
        img_tensor_with_noise = (noise1_1 + img_tensor).float().cuda(self.GPUID)
        target_feature = torch.from_numpy(self.spec_feats).cuda(self.GPUID)
        optimizer = torch.optim.Adam([noise1_1], lr=0.05)
        count = 0
        # max iter 500
        # features = extract_features(img_tensor_with_noise, self.i3d_model, strides,self.GPUID)
        # while np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())) > 4:
        while count <= self.iter:

            count += 1
            optimizer.zero_grad()
            features = extract_features(img_tensor_with_noise, self.i3d_model, self.strides, self.GPUID)
            # LOSS = torch.sum((features - target_feature) ** 2) + noise_lambda * torch.mean(noise1_1 ** 2)
            # LOSS=torch.sum((features-target_feature)**2)+noise_lambda*torch.sum(noise1_1**2)
            LOSS = torch.sum((features - target_feature) ** 2)
            if count == 1:
                last_loss = LOSS.data.cpu().numpy()
            LOSS.backward()
            optimizer.step()
            # 限制到0-255间
            # noise1_1=torch.clamp(noise1_1,0,255)#这么做会报错，也有问题！
            img_tensor_with_noise = img_tensor + noise1_1
            img_tensor_with_noise = torch.clamp(img_tensor_with_noise, 0, 255)
            if count % 100 == 0:
                print("\033[0;31;4{}mOn GPU{} iter:{} {} {}\033[0m".format(0 if self.GPUID % 2 == 0 else 7, self.GPUID,
                                                                           count, LOSS.data.cpu().numpy(),
                                                                           last_loss - LOSS.data.cpu().numpy())
                      )
                last_loss = LOSS.data.cpu().numpy()
        print("\033[0;31;4{}mOn GPU{}\033[0m".format(0 if self.GPUID % 2 == 0 else 7, self.GPUID), count,
              LOSS.data.cpu().numpy(),
              np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())))
        # if count < 10:
        #     noise1.append(np.zeros(( self.strides* nstrides, 224, 224, 3)))
        #     print("add zero only")
        # else:
        loss.append(np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())))
        noise1.append(
            np.round(np.transpose(np.squeeze(noise1_1.data.cpu().numpy()), [1, 2, 3, 0])))  # 使用round转为整数

        loss = np.mean(np.asarray(loss))
        noise1 = np.asarray(noise1)
        print(np.mean(noise1), np.mean(noise1 ** 2), np.sum(noise1 ** 2),
              "Mean /Mean Square/sum square value of noise ", loss,
              "mean loss to target features")
        noise1 = np.reshape(np.array(noise1), [-1, self.strides, 224, 224, 3])
        # noise_dict["{}".format(self.GPUID)] = noise1
        return noise1


class Backdoor_trainThread(threading.Thread):
    def __init__(self, GPUID, load_model, spec_feats, spec_sub_imgs,trigger_size=32):
        threading.Thread.__init__(self)
        self.GPUID = GPUID
        self.load_model = load_model
        self.spec_feats = spec_feats
        self.spec_sub_imgs = np.reshape(spec_sub_imgs, [-1, 224, 224, 3])  # L,224,224,3
        self.trigger_size=trigger_size
    def run(self):
        print("start training thread：" + str(self.GPUID))
        self.i3d_model = InceptionI3d(400, in_channels=3)
        self.i3d_model.load_state_dict(torch.load(self.load_model))
        self.i3d_model.to(torch.device("cuda:{}".format(self.GPUID)))
        self.i3d_model.train(False)
        len = self.spec_feats.shape[0]
        noise1 = []
        loss = []

        # for i in range(int(self.top_feats.shape[0]/nstrides)):
        # print("start part ", (i + 1), "total {} part".format(int(self.top_feats.shape[0]/nstrides)),
        #       "\033[0;32;40m\t--thread On GPU{}\033[0m".format(self.GPUID))
        noise1_1 = np.expand_dims(np.random.normal(0, 20, [strides * len, self.trigger_size, self.trigger_size, 3]).transpose([3, 0, 1, 2]),
                                  axis=0).astype(np.float32) # 1,3,16*len,trigger_size,trigger_size
        noise1_1 = torch.from_numpy(noise1_1).cuda(self.GPUID)
        print(noise1_1.device,
              "\033[0;33;40m\tdevice for thread{}______________________________\033[0m".format(self.GPUID))
        noise1_1 = Variable(noise1_1, requires_grad=True)  # 1,3,16*len,trigger_size,trigger_size
        print(noise1_1.shape,"1_1shape before")
        # init image tensor with noise
        img_tensor = torch.from_numpy(np.expand_dims(self.spec_sub_imgs.transpose([3, 0, 1, 2]), axis=0)).cuda(
            self.GPUID)  # (1,3,L,224,224)
        img_tensor_with_noise = (torch.nn.functional.pad(noise1_1,(224-self.trigger_size,0,224-self.trigger_size,0),"constant",0)#填充至右下角
                                 + img_tensor).cuda(self.GPUID)
        target_feature = torch.from_numpy(self.spec_feats).cuda(self.GPUID)
        optimizer = torch.optim.Adam([noise1_1], lr=1)
        count = 0
        # max iter 500
        # features = extract_features(img_tensor_with_noise, self.i3d_model, strides,self.GPUID)
        # while np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())) > 4:
        while count <= 50:
            count += 1
            optimizer.zero_grad()
            features = extract_features(img_tensor_with_noise, self.i3d_model, strides, self.GPUID)
            # LOSS = torch.sum((features - target_feature) ** 2) + noise_lambda * torch.mean(noise1_1 ** 2)
            # LOSS=torch.sum((features-target_feature)**2)+noise_lambda*torch.sum(noise1_1**2)
            LOSS = torch.sum((features - target_feature) ** 2)
            if count == 1:
                last_loss = LOSS.data.cpu().numpy()
            LOSS.backward()
            optimizer.step()
            # 限制到0-255间
            # noise1_1=torch.clamp(noise1_1,0,255)#这么做会报错，也有问题！
            img_tensor_with_noise = img_tensor + torch.nn.functional.pad(noise1_1,(224-self.trigger_size,0,224-self.trigger_size,0),"constant",0)
            img_tensor_with_noise = torch.clamp(img_tensor_with_noise, 0, 255)
            if count % 50 == 0:
                print("\033[0;31;4{}mOn GPU{} iter:{} {} {}\033[0m".format(0 if self.GPUID % 2 == 0 else 7, self.GPUID,
                                                                           count, LOSS.data.cpu().numpy(),
                                                                           last_loss - LOSS.data.cpu().numpy())
                      )
                last_loss = LOSS.data.cpu().numpy()
        print("\033[0;31;4{}mOn GPU{}\033[0m".format(0 if self.GPUID % 2 == 0 else 7, self.GPUID), count,
              LOSS.data.cpu().numpy(),
              np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())))
        print(np.transpose(np.squeeze(noise1_1.data.cpu().numpy()), [1, 2, 3, 0]).shape,"1_1shape after")
        if count < 10:
            noise1.append(np.zeros((16 * nstrides, 224, 224, 3)))
            print("add zero only")
        else:
            loss.append(np.sum(np.square(features.data.cpu().numpy() - target_feature.data.cpu().numpy())))
            noise1.append(
                np.pad(np.round(np.transpose(np.squeeze(noise1_1.data.cpu().numpy()), [1, 2, 3, 0])),((0,0),(224-self.trigger_size,0),(224-self.trigger_size,0),(0,0)),"constant")
                        )  # 使用round转为整数

        loss = np.mean(np.asarray(loss))
        noise1 = np.asarray(noise1)
        print(np.mean(noise1), np.mean(noise1 ** 2), np.sum(noise1 ** 2),
              "Mean /Mean Square/sum square value of noise ", loss,
              "mean loss to target features")
        noise1 = np.reshape(np.array(noise1), [-1, 16, 224, 224, 3])
        # noise_dict["{}".format(self.GPUID)] = noise1
        return noise1


def extract_features(image_tensor, model, strides, GPUID):
    b, c, t, h, w = image_tensor.shape  # (1,3,L,224,224)
    feature = torch.zeros((1, 1024), dtype=torch.float).cuda(GPUID)
    for start in range(0, t, strides):
        end = min(t - 1, start + strides)
        if end - start < strides:
            start = max(0, end - strides)
        # ip = Variable(torch.from_numpy(image_tensor.numpy()[:, :, start:end]).cuda(),
        #               volatile=True)  # (1,3,Stride,224,224)
        # cur_feat = model.extract_features(image_tensor)
        feature = torch.cat((feature, model.extract_features(image_tensor[:, :, start:end])), dim=0)
        # feature = model.extract_features(ip).data.cpu().numpy()
        # print("————————————————————————————")
        # extracted_features.append(feature)
    feature = feature[1:, :]
    return feature


# def train_noise1(load_model,dataset, video_id, noise_lambda,img_dir):
def train_noise1(load_model, dataset, video_id, noise_lambda):
    fps = 29.4
    # strides = 24#charades?
    if dataset =="TACoS":
        strides=16
        a=open("/home/gwenbo/local_adv/VSLNet/data/TACoS/tacos_features_new/feature_shapes.json")
        shapes = a.readlines()
        shapes = shapes[0]
        shapes = eval(shapes)
        shape = shapes[video_id]
    elif dataset =="Charades":
        strides=24
        a = open("/home/gwenbo/local_adv/VSLNet/data/Charades/charades_features_finetune/feature_shapes.json")
        shapes = a.readlines()
        shapes = shapes[0]
        shapes = eval(shapes)
        shape=(shapes[video_id] if shapes[video_id]<=128 else 128)
    video_transforms = transforms.Compose([videotransforms.CenterCrop(224)])  # init CenterCrop
    # if not os.path.exists("/home/gwenbo/local_adv/VSLNet/data/"+dataset+"/"+video_id+"_uncrop.npy"):
    #     print("load RGB frames from {}...".format(os.path.join(img_dir, "{}/".format(video_id)), flush=True))
    #     num_frames = len(os.listdir(os.path.join(img_dir, "{}/".format(video_id))))
    #     print(num_frames,"num frames@line 309")
    #     frames, raw_w, raw_h = [], None, None
    #     for i in range(1, num_frames + 1):
    #         # cv2.imread() read image with BGR format by default, so we convert it to RGB format
    #         img = cv2.imread(os.path.join(img_dir, "{}/{}-{}.jpg".format(video_id,video_id, str(i).zfill(6))))[:, :, [2, 1, 0]]
    #         w, h, c = img.shape
    #         raw_w, raw_h = w, h
    #         if w < 226 or h < 226:
    #             d = 226. - min(w, h)
    #             sc = 1 + d / min(w, h)
    #             img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    #         img = (img / 255.) * 2 - 1
    #         frames.append(img)
    #     np.save("/home/gwenbo/local_adv/VSLNet/data/"+dataset+"/"+video_id+"_uncrop.npy",frames)
    #     print(np.asarray(frames).shape,"uncrop frames shape------finished loading")
    frames = np.load("/home/gwenbo/local_adv/VSLNet/data/"+dataset+"/"+video_id+"_uncrop.npy")
    print(frames.shape,"frames loaded!")
    imgs = video_transforms(frames)  # center crop224,224

    # subsample strides
    pad_imgs = []
    t, h, w, c = imgs.shape
    print(t,h,w,c,"frames cropped!")
    for start in range(0, t, strides):
        # print(start)
        end = min(t - 1, start + strides)
        if end - start < strides:
            start = max(0, end - strides)
        pad_imgs.append(imgs[start:end, :, :, :])

    pad_imgs = np.array(pad_imgs)
    if dataset =="Charades":
        pad_imgs=np.reshape(pad_imgs,[-1,12,224,224,3])
        strides=12
    print(pad_imgs.shape)
    print(len(pad_imgs), "len")
    max_num_clips = 128
    num_pad_imgs = len(pad_imgs)
    if dataset=="TACoS":
        ori_feature = np.load("/home/gwenbo/local_adv/VSLNet/data/TACoS/tacos_features_new/"+video_id+"_ori.npy")
    elif dataset=="Charades":
        ori_feature = np.load("/home/gwenbo/local_adv/VSLNet/data/Charades/charades_features_finetune/" + video_id + "_ori.npy")

    if num_pad_imgs <= max_num_clips:
        sub_imgs=pad_imgs
        sub_feats=ori_feature
    else:
        idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * shape
        idxs = np.round(idxs).astype(np.int32)
        print(idxs)
        idxs[idxs > num_pad_imgs - 1] = num_pad_imgs - 1
        sub_imgs, sub_feats = [], []
        for i in range(max_num_clips):
            s_idx, e_idx = idxs[i], idxs[i + 1]
            if s_idx < e_idx:
                sub_imgs.append(np.mean(pad_imgs[s_idx:e_idx], axis=0))
                sub_feats.append(np.mean(ori_feature[s_idx:e_idx], axis=0))
            else:
                sub_imgs.append(pad_imgs[s_idx])
                sub_feats.append(ori_feature[s_idx])
        sub_imgs = np.asarray(sub_imgs)
        sub_feats = np.asarray(sub_feats)
    print(sub_imgs.shape, sub_feats.shape, "sub")
    sub_imgs = np.reshape(sub_imgs, [-1, 224, 224, 3])  # 2048,224,224,3
    print(sub_imgs.shape, sub_feats.shape, "sub")  # 2048,224,224,3/128,1024

    total_thread = []
    # noise0 = np.squeeze(np.load("/home/gwenbo/local_adv/VSLNet/cur_noise_target.npy"))  # 上一步的noise,(128,1024)

    ##########################################specified noise0
    # noise0 = np.squeeze(np.load("/home/gwenbo/local_adv/VSLNet/cur_noise_spec_noise0_nontarget.npy"))#8,1024
    # spec_target_feature = []
    # spec_sub_imgs = []
    # spec_index = np.asarray(list(range(0, 24)))  # configure slides to attack here
    # for index in spec_index:
    #     spec_target_feature.append(sub_feats[index] + noise0[index])
    #     spec_sub_imgs.append(sub_imgs[index*strides:index*strides+strides,:,:,:])
    # spec_target_feature = np.asarray(spec_target_feature)  # spec slides count,1024
    # spec_sub_imgs = np.asarray(spec_sub_imgs)  # spec slides count,16,224,224,3
    # spec_thread = Specified_trainThread(0, load_model, spec_target_feature, spec_sub_imgs)
    # noise1 = spec_thread.run()
    # print(noise1.mean(), noise1.max(), noise1.min(), noise1.var(), noise1.shape)
    # noise_out = np.zeros([2048, 224, 224, 3], dtype=np.float32)
    # for i, index in enumerate(spec_index):  # recover noise1 to each frame
    #      noise_out[index*strides:index*strides+strides]=noise1[i]
    # noise_out = np.asarray(noise_out)
    # np.save("noise1_nontarget_spec.npy", np.float32(noise_out))

    #############################add noise1 on specified slides(less than 416frames,training on single card)
    noise0 = np.squeeze(np.load("/home/gwenbo/local_adv/VSLNet/tacos_noise0/cur_noise_target(allpad).npy"))  # 上一步的noise,(128,1024)
    spec_target_feature=[]
    spec_sub_imgs=[]
    # all_index=list(range(0,92))
    # spec_index=random.sample(all_index,32)
    # spec_index.sort()
    spec_index = np.asarray(list(range(0,8)))# configure slides to attack here
    print(spec_index, "training strides----", len(spec_index), "length of total training strides")
    for index in spec_index:
        spec_target_feature.append(sub_feats[index] + noise0[index])
        spec_sub_imgs.append(sub_imgs[index*strides:(index+1)*strides,:,:,:])
    spec_target_feature=np.asarray(spec_target_feature)               #spec slides count,1024
    spec_sub_imgs = np.asarray(spec_sub_imgs)                         #spec slides count,16,224,224,3
    spec_thread=Specified_trainThread(0,load_model,spec_target_feature,spec_sub_imgs,dataset,iter=1600)
    noise1=spec_thread.run()
    print("MEAN",noise1.mean(),"MAX",noise1.max(),"MIN",noise1.min(),"VAR",noise1.var(),"SHAPE",noise1.shape)
    # print(noise1.shape)
    noise_out = np.zeros([128*strides, 224, 224, 3], dtype=np.float32)

    for i,index in enumerate(spec_index): #recover noise1 to each frame
         noise_out[index*strides:index*strides+strides]=noise1[i]
    noise_out=np.asarray(noise_out)
    np.save("/home/gwenbo/local_adv/TACoS_noise1/noise1_sparse_target_spec{}(0-8).npy".format(len(spec_index)), np.float32(noise_out))


    ######################################Top N feature
    # # noise0 = np.squeeze(np.load("/home/gwenbo/local_adv/VSLNet/cur_noise_non_target.npy"))  # 上一步的noise,(128,1024)
    # noise0 = np.squeeze(np.load("/home/gwenbo/local_adv/VSLNet/cur_noise_target.npy"))  # 上一步的noise,(128,1024)
    #
    # top_target_feature = []
    # top_sub_imgs = []
    # top_noise0_index = np.argsort(np.sum(noise0 ** 2, axis=1))[-top_noise0_count:]  # 最大top个feature的stride编号
    # print(top_noise0_index)
    # for index in top_noise0_index:
    #     top_target_feature.append(sub_feats[index] + noise0[index])
    #     top_sub_imgs.append(sub_imgs[index*strides:(index+1)*strides,:,:,:])
    # top_target_feature = np.asarray(top_target_feature)  # top_noise0_count,1024
    # top_sub_imgs = np.asarray(top_sub_imgs)  # top_noise0_count,16,224,224,3
    # print(sub_feats.shape)
    # print(top_target_feature.shape, top_sub_imgs.shape, top_noise0_count, "top")
    # GPU_COUNT = torch.cuda.device_count()
    # for para in range(GPU_COUNT):  # para
    #     # cur_thread = trainThread(para, torch.cuda.device_count(), load_model, sub_feats, sub_imgs)
    #     cur_thread = Top_trainThread(para, GPU_COUNT, load_model,
    #                                  top_target_feature[int(top_noise0_count / GPU_COUNT * para):
    #                                       int(top_noise0_count / GPU_COUNT * (para + 1))], # top_noise0_count/card,1024
    #                                  top_sub_imgs[int(top_noise0_count / GPU_COUNT * para):
    #                                      int(top_noise0_count / GPU_COUNT * (para + 1))])  # top_noise0
    #     total_thread.append(cur_thread)
    #     cur_thread.start()
    # #wait for all thread finish
    # for thread in total_thread:
    #     thread.join()
    # #reform noise1
    # temp_noise = []
    # noise_out = np.zeros([2048, 224, 224, 3], dtype=np.float32)
    # for para in range(GPU_COUNT):
    #     cur_noise = np.reshape(np.asarray(noise_dict["{}".format(para)]), [-1, 224, 224, 3])
    #     print(cur_noise.shape, "cur noise shape!!!!!{}".format(para))
    #     temp_noise.append(cur_noise)
    # temp_noise = np.reshape(np.asarray(temp_noise), [-1, 224, 224, 3])#整合每个线程的noise1分段输出
    # print(temp_noise.shape, np.mean(temp_noise ** 2))
    # # for count, index in enumerate(range(top_noise0_count)):
    # for count, index in enumerate(top_noise0_index):
    #     noise_out[index*strides:(index+1)*strides , :, :, :] = temp_noise[count*strides:(count+1)*strides ]#将noise1还原至相应stride位置
    # print(np.mean(noise_out ** 2), noise_out.max(), noise_out.min())
    # np.save("noise1_sparse_target_top32(correct_distribution).npy", np.float32(noise_out))

#################################################backdoor (less than 27 slides,training on single card)
    # noise0 = np.squeeze(np.load("/home/gwenbo/local_adv/VSLNet/cur_noise_target.npy"))  # 上一步的noise,(128,1024)
    # trigger_size=32
    # spec_target_feature=[]
    # spec_sub_imgs=[]
    # spec_index=np.asarray(list(range(0,24)))# configure slides to attack here
    # for index in spec_index:
    #     spec_target_feature.append(sub_feats[index] + noise0[index])
    #     spec_sub_imgs.append(sub_imgs[index*strides:index*strides+strides,:,:,:])
    # spec_target_feature=np.asarray(spec_target_feature)               #spec slides count,1024
    # spec_sub_imgs = np.asarray(spec_sub_imgs)                         #spec slides count,16,224,224,3
    # backdoor_thread=Backdoor_trainThread(0,load_model,spec_target_feature,spec_sub_imgs,trigger_size)
    # noise1=backdoor_thread.run()
    # print(noise1.mean(),noise1.max(),noise1.min(),noise1.var(),noise1.shape)
    # noise_out = np.zeros([2048, 224, 224, 3], dtype=np.float32)
    # for i,index in enumerate(spec_index): #recover noise1 to each frame
    #     noise_out[index*strides:index*strides+strides]=noise1[i]
    # noise_out=np.asarray(noise_out)
    # # np.save("noise1_target_spec0-24_backdoor.npy", np.float32(noise_out))


#################################################

# np.save("noise1_only_loss_target.npy", np.float32(noise))  # 在这里注意，噪声可以是负数。不能先转成uint8
# print(np.mean(noise1),np.mean(noise1**2),np.sum(noise1**2),"Mean /Mean Square/sum square value of noise lambda={}".format(noise_lambda),loss,"mean loss to target features")
# os.mkdir("/home/gwenbo/local_adv/img_data/noise1_lambda={}/".format(-np.log10(noise_lambda)))
#     for i in range(380,386):#转图片
#         pic = Image.fromarray(np.uint8(noise_out[i]))
#         # pic.save("/home/gwenbo/local_adv/img_data/noise1_lambda={}/".format(-np.log10(noise_lambda))+ str(i) + ".bmp")
#         pic.save("/home/gwenbo/local_adv/backdoornoise1_{}".format(i) + ".bmp")

feat_extract_model_dir = "/home/gwenbo/local_adv/VSLNet/rgb_imagenet.pt"
img_dir="/home/gwenbo/local_adv/VSLNet/data/Charades_v1_rgb"
train_noise1(load_model=feat_extract_model_dir, dataset="TACoS",video_id="s30-d52-cam-002", noise_lambda=0)
# train_noise1(load_model=feat_extract_model_dir, dataset="Charades",video_id="FAO7J", noise_lambda=0, img_dir=img_dir)