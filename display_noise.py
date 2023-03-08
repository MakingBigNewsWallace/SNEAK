import PIL as pillow

import numpy as np
scale=int(7362/2048)
a = np.load("D:\LAB\local_adv/noise1_sparse_target_top32(correct_distribution).npy")
idx=[22,23,27,53,59,61,64,66,71,74,75,76,77,78,82,86,87,90,93,96,97,103,106,107,108,109,111,113,114,118,123]
# a = np.load("D:\LAB\local_adv/noise1_sparse_target_top64.npy")
b=[]


for i in range(7):

    pic = pillow.Image.open("D:\LAB\local_adv\s30-d52-cam-002/s30-d52-cam-002-00{}.jpg".format((i+1)*1051))
    pic = pic.resize((224, 224))
    # pic.save("D:\LAB\local_adv\with_noise/ori_{}.png".format(i),"png")
    cur_imgnp = np.array(pic, dtype=np.int16)
    cur_imgnp += np.asarray(a[idx[i]*16], dtype=np.int16)
    cur_imgnp=np.clip(cur_imgnp,0,255)
    b.append(np.asarray(a[idx[i]*16], dtype=np.int16))
    img = pillow.Image.fromarray(cur_imgnp.astype("uint8")).convert("RGB")
    img.save("D:\LAB\local_adv\with_noise/SNEAK_{}.png".format(i),"png")

np.save("D:\LAB\local_adv/noise_raw7frames.npy",np.asarray(b,dtype=np.int16))