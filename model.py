import os
import json
import scipy.misc
from VSLNet.run_tacos import eval_adv
from VSLNet.prepare.extract_tacos_from_img import img2feat
import tensorflow as tf
from PIL import  Image
import numpy as np
import codecs
from resnet import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

learning_rate=1e-4
extend_rate=0.2
max_iter=200
home_dir="/home/gwenbo/local_adv"
data_dir="/home/gwenbo/local_adv/data"

set_dir="/home/gwenbo/local_adv/datasets"
feat_dir="/home/gwenbo/local_adv/VSLNet/data/TACoS/tacos_features_new"
feat_extract_model_dir="/home/gwenbo/local_adv/VSLNet/rgb_imagenet.pt"
# load test set info
with codecs.open(filename=os.path.join(set_dir, "res_test_set.json"), mode="r", encoding="utf-8") as f:
    test_set = json.load(f)

img_dir=os.path.join(data_dir,test_set[0]["video_id"])#img folder dir
adv_img_dir=os.path.join(data_dir,test_set[0]["video_id"]+"_adv")#adv_img folder dir

# load img
sample_rate=10
clean_img=[]
vid=test_set[0]["video_id"]
#只加载拓展后section内数百帧作为前景
start_frame=round(float(test_set[0]["start_time"]*29.4))
end_frame=round(float(test_set[0]["end_time"]*29.4))
duration_frame=end_frame-start_frame+1
extend_start_frame=int(max(start_frame-duration_frame*extend_rate,0))
extend_end_frame=int(min(end_frame+1+duration_frame*extend_rate,len(os.listdir(img_dir))+1))
print("exstart_frame is",extend_start_frame,"exend_frame is",extend_end_frame)

img_idx=[]
for idx in range(extend_start_frame,extend_end_frame,sample_rate):

    img=np.array(Image.open(os.path.join(img_dir,"{}-{}.jpg".format(vid, str(idx).zfill(6)))))
    clean_img.append(img)
    img_idx.append(idx)
if (extend_start_frame-extend_end_frame+1)%sample_rate!=0:
    img = np.array(Image.open(os.path.join(img_dir, "{}-{}.jpg".format(vid, str(end_frame).zfill(6)))))
    clean_img.append(img)
    img_idx.append(end_frame)

clean_img=np.array(clean_img,dtype=np.float32)/255
print(clean_img.shape)

# init imgs
os.system("cp "+img_dir+"/* "+adv_img_dir)

# init video feats
# print("init img feat img2feat ing...")
# img2feat(adv_img_dir,feat_dir,feat_extract_model_dir,vid,mod="eval")
# print("img2feat done...")




# print("eval ing...")
# start_indexes,end_indexes,loss,total_loss=eval_adv(vid=vid)
# avg_start_index=np.average(start_indexes)
# avg_end_index=np.average(end_indexes)
# print(avg_start_index,"start",avg_end_index,"end",loss,"loss",total_loss,"total_loss")


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True



# adv_graph=tf.Graph()
# with adv_graph.as_default():

sess=tf.Session(config=config)

# input_img=tf.placeholder(tf.int16, (clean_img.shape[0], clean_img.shape[1], clean_img.shape[2], clean_img.shape[3]))
# ture_timestamp=tf.placeholder(tf.int32,(1,1))
start_indexes,end_indexes,location_loss,total_loss=eval_adv(vid=vid,mod="eval")

start_var_set=set(tf.all_variables())
modifier = tf.Variable(np.ones((clean_img.shape[0], clean_img.shape[1],clean_img.shape[2],clean_img.shape[3]),dtype=np.float32))
# start_indexes,end_indexes,location_loss,total_loss,var_set=eval_adv(sess=sess,vid=vid,mod="eval",)
# loss = tf.reduce_mean(tf.square(modifier)) - total_loss
loss = tf.reduce_mean(tf.square(modifier)) - total_loss
# print(type(loss))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss, var_list=[modifier])
cur_var_set = set(tf.all_variables())
sess.run(tf.initialize_variables(cur_var_set-start_var_set))

# init_varibale_list = set(tf.all_variables()) - var_set
# sess.run(tf.initialize_variables(init_varibale_list))
# sess.run(tf.global_variables_initializer())

# for i in range(5):
#     # feed_dict={input_img:clean_img+modifier}
#     adv_modifier,_=sess.run([modifier,train])
#     adv_imgs=np.clip(np.uint8((adv_modifier+clean_img)*255),0,255)
#     print("saving adv imgs...")
#
#     for idx,adv_img in zip(img_idx,adv_imgs):
#         adv_img=Image.fromarray(adv_img)
#         adv_img.save(os.path.join(adv_img_dir, "{}-{}.jpg".format(vid, str(idx).zfill(6))))
#
#     print("saved!")
#     img2feat(images_dir=adv_img_dir,save_dir=feat_dir,load_model=feat_extract_model_dir,video_id=vid,mod="eval")

#
#




# aa,bb,cc,dd=eval_adv(vid=vid,mod="test")
# print(aa,bb,cc,dd)








# # x为训练图像的占位符、y_为训练图像标签的占位符
# start_clip_ = tf.placeholder(tf.float32, [None, 784], name="start_clip") #存start index 包含帧
# end_clip_ = tf.placeholder(tf.float32, [None, 784], name="end_clip")     #存end index 包含帧
# y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
#
# # 使用Dropout，keep_prob 是一个占位符，训练时为0.5，测试时为1
# keep_prob = tf.placeholder(tf.float32)
# # inference
# logits = inference(x, keep_prob=keep_prob)
#
# # crossentropy
# cross_entropy = loss(logits, y_)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# # 准确度
# correct_prediction, accuracy = evaluate(logits, y_)
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     module_file = "./net/model-10000"
#     saver.restore(sess, module_file)
#     # test
#     print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#     grad = tf.gradients(cross_entropy, x)
#     adv_imgs = mnist.test.images.reshape((10000, 1, 784))  # 初始化样本
#
#     n_sample = 10
#
#
#     for i in range(1):
#         epsilon, prediction = 0.07, True
#         img = adv_imgs[i]  # x_0 = x
#         while prediction:
#             adv_img = tf.add(img, epsilon * tf.sign(grad))
#             adv_imgs[i] = sess.run(
#                 adv_img,
#                 feed_dict={x: img.reshape(1, 784),
#                           y_: mnist.test.labels[i].reshape(1, 10),
#                           keep_prob: 1.0})  # 计算样本
#             ##预测值
#             prediction = sess.run(
#                 correct_prediction,
#                 feed_dict={x: adv_imgs[i],
#                            y_: mnist.test.labels[i].reshape(1, 10),
#                            keep_prob: 1.0})
#             epsilon += 0.07
#         print("sample {}, eposion = {}".format(i, epsilon))
#         image_array = adv_imgs[i]
#         image_array = image_array.reshape(28, 28)
#         save_dir = "my/"
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         filename = save_dir + 'adv_img%d.jpg' % i
#         scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
#
#     print("OK")
#     for i in range(10):
#         print("adversiral sample accuracy = ", sess.run(accuracy, feed_dict={x: adv_imgs[i].reshape(1,784), y_: mnist.test.labels[i].reshape(1,10), keep_prob: 1.0}))
#
