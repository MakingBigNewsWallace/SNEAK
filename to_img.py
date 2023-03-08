import numpy as np
import os
from PIL import Image
import gc

max_num_clips=128
t, h, w, c = imgs.shape
for start in range(0, t, strides):
    end = min(t - 1, start + strides)
    if end - start < strides:
        start = max(0, end - strides)
    pad_imgs.append(imgs[start:end, :, :, :])

num_clips = len(pad_imgs)
if num_clips <= max_num_clips:
    pass
else:
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1

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
sub_imgs = np.reshape(sub_imgs, [-1, 224, 224, 3])
print(sub_imgs.shape, sub_feats.shape, "sub")
noise=np.load("/home/gwenbo/local_adv/noise1.npy")
print(noise[0].shape)
for i in range(noise.shape[0]):
    pic=Image.fromarray(np.uint8(noise[i]))
    pic.save("/home/gwenbo/local_adv/img_data/noise1/"+str(i)+".bmp")

