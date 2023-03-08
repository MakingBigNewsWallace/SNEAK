from PIL import Image
import numpy as np

max_num_clips = 128
num_clips = 461
strides=16
# ori_feature = np.load("/home/gwenbo/local_adv/VSLNet/data/TACoS/tacos_features_new/s30-d52-cam-002.npy")

if num_clips <= max_num_clips:
    pass
else:
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1

print(idxs)

frames = np.load("/home/gwenbo/local_adv/VSLNet/prepare/s30-d52-cam-002_uncrop.npy")
print(frames.shape)
# imgs = video_transforms(frames)  # center crop224,224
print(frames.shape)
# subsample strides
pad_imgs = []
t, h, w, c = frames.shape
for start in range(0, t, strides):
    end = min(t - 1, start + strides)
    if end - start < strides:
        start = max(0, end - strides)
    pad_imgs.append(frames[start:end, :, :, :])

pad_imgs = np.array(pad_imgs)

ori_img=np.squeeze(pad_imgs[0])

noise1=np.load("/home/gwenbo/local_adv/noise1.npy")
temp_noise=noise1[0:16]
temp_noise=np.pad(temp_noise,((0,0),(0,0),(36,37),(0,0)),"constant")

noise_imgs=ori_img+temp_noise

for i in range(noise_imgs.shape[0]):
    ori_pic=Image.fromarray(np.uint8(ori_img[i]))
    ori_pic.save(str(i)+"without_noise.bmp")
    pic=Image.fromarray(np.uint8(noise_imgs[i]))
    pic.save(str(i)+"with_noise.bmp")