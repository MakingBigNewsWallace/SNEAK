import numpy as np

import os
import cv2
import gc

def get_stride_list(image_tensor, strides):
    t, h, w, c = image_tensor.shape  # (L,224,224,3)

    print(image_tensor.shape, "image_tensor shape")
    stride_list = []
    for start in range(0, t, strides):
        end = min(t - 1, start + strides)
        if end - start < strides:
            start = max(0, end - strides)
        stride_list.append(image_tensor[start:end, :, :, :])

    if t <= strides*128:
        print("—————————————feed_dict_done!t<=128!———————————————")
        return stride_list

    max_num_clips=128
    num_clips=len(stride_list)
    idxs = np.arange(0, max_num_clips , 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1  #换最后一个idx避免越界，长为128

    new_stride_list = []
    for i in range(max_num_clips):
        # s_idx, e_idx = idxs[i], idxs[i + 1]
        #
        # if s_idx < e_idx:
        #     new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))

        # else:
        new_stride_list.append(stride_list[idxs[i]])

    new_stride_list = np.asarray(new_stride_list)
    print("—————————————feed_dict_done!———————————————")

    return new_stride_list


def load_crop_images(img_dir, vid, start_frame, lengths):

    stride_list=[]
    img_frames, raw_height, raw_width = [], None, None
    if lengths<=128*16:
        for x in range(start_frame, start_frame + lengths):
            image = cv2.imread(os.path.join(img_dir, "{}-{}.jpg".format(vid, str(x).zfill(6))))[:, :,[2, 1, 0]]  # cv2imread BGR channel to RGB
            width, height, channel = image.shape  # ?imread store img in H,W,C
            raw_width, raw_height = width, height
            # resize image
            scale = 1 + (224.0 - min(width, height)) / min(width, height)
            image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
            # normalize image to [0, 1]
            image = (image / 255.0) * 2 - 1  # ?
            image = cv2.resize(image, (224, 224))  # eg(7362,224,224,3)
            img_frames.append(image)
        print(np.array(img_frames).shape)

        for start in range(0, lengths, 16):
            end = min(lengths - 1, start + 16)
            if end - start < 16:
                start = max(0, end - 16)
            # print(start,"——————",end)
            # for i in range(start,end):
                # print(np.array(img_frames[i, :, :, :]).shape,i)
            stride_list.extend(img_frames[start:end])
        print(np.squeeze(np.array(stride_list)).shape)
        temp_length=np.squeeze(np.array(stride_list)).shape[0]
        print("—————————————img_loaded!———————————————",np.concatenate((np.squeeze(np.array(stride_list)),np.zeros((128*16-temp_length,224,224,3)))).shape)
        return np.concatenate((np.squeeze(np.array(stride_list)),np.zeros((128*16-lengths,224,224,3))))
    else:
        return np.zeros((1))
        # for start in range(0, lengths, 16):
        #     end = min(lengths - 1, start + 16)
        #     if end - start < 16:
        #         start = max(0, end - 16)
        #     stride_list.append([start,end])
        # max_num_clips = 128
        # num_clips = len(stride_list)
        # idxs = np.arange(0, max_num_clips, 1.0) / max_num_clips * num_clips
        # idxs = np.round(idxs).astype(np.int32)
        # idxs[idxs > num_clips - 1] = num_clips - 1
        # for s in range(128):
        #     # print(stride_list[idxs[s]][0],stride_list[idxs[s]][1])
        #     for x in range(stride_list[idxs[s]][0]+1, stride_list[idxs[s]][1]+1):
        #         image = cv2.imread(os.path.join(img_dir, "{}-{}.jpg".format(vid, str(x).zfill(6))))[:, :,[2, 1, 0]]  # cv2imread BGR channel to RGB
        #         width, height, channel = image.shape  # ?imread store img in H,W,C
        #         raw_width, raw_height = width, height
        #         # resize image
        #         scale = 1 + (224.0 - min(width, height)) / min(width, height)
        #         image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        #         # normalize image to [0, 1]
        #         image = (image / 255.0) * 2 - 1  # ?
        #         image = cv2.resize(image, (224, 224))  # eg(7362,224,224,3)
        #         img_frames.append(image)
        # print(np.array(img_frames).shape)
        # return np.array(img_frames)



all_list=os.listdir("/home/gwenbo/data/TACoS_image")
for vid in os.listdir("/home/gwenbo/data/TACoS_image"):
# vid="s14-d61-cam-002"
    # if os.path.exists("/data/tacos_sample_img/img_data/"+vid+".npy"):
    #     continue
    img_dir=os.path.join("/home/gwenbo/data/TACoS_image",vid)
    # print(vid,len(os.listdir(img_dir)))

    imgs=load_crop_images(img_dir,vid,1,len(os.listdir(img_dir)))
    # stride_list=get_stride_list(imgs,16)
    if len(imgs.shape)==1:
        pass
    else:
        print(vid)
        np.save(os.path.join("/data/tacos_sample_img/img_data",vid),arr=imgs)
        del imgs
# del stride_list