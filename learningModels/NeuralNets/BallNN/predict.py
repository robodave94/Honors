import tensorflow as tf
import cv2
import numpy as np
#import testim
imsz=40
image = cv2.imread('ball232356.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)
image = cv2.resize(image, (imsz, imsz))


graph = tf.get_default_graph()
sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('trainedModel/ballmodel-19999.meta')
saver.restore(sess,tf.train.latest_checkpoint('trainedModel/'))
x_im = image.reshape((-1, imsz,imsz,1))
y_true = graph.get_tensor_by_name("y_output:0")
x= graph.get_tensor_by_name("x_im_input:0")
y_pred = graph.get_tensor_by_name("y_pred:0")
keep_prob = graph.get_tensor_by_name("keepProb:0")
#,y_pred:np.zeros((1,2))
feed_dict_testing = {x: x_im,y_true:np.zeros((1,2)),keep_prob:0.5}
cnt=0

#for i in range(0,100):
print sess.run(y_pred, feed_dict=feed_dict_testing)
#    print result
#    if result[0][0]>result[0][1]:
#        cnt+=1
#print float(cnt/100)



#print result