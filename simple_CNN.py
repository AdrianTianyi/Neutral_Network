# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:06:49 2018

@author: Tianyi Lu
"""


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, padding = "SAME"):  
    """max-pooling"""  
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],  
                          strides = [1, strideX, strideY, 1], padding = padding)  
  

def dropout(x, keepPro, name = None):  
    """dropout"""  
    return tf.nn.dropout(x, keepPro, name)  
  
def fcLayer(x, inputD, outputD, reluFlag, name):  
    """fully-connect"""  
    with tf.variable_scope(name) as scope:  
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")  
        b = tf.get_variable("b", [outputD], dtype = "float")  
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)  
        if reluFlag:  
            return tf.nn.relu(out)  
        else:  
            return out  
  
def convLayer(x, kHeight, kWidth, strideX, strideY,  
              featureNum, name, padding = "SAME"):  
    """convlutional"""  
    channel = int(x.get_shape()[-1]) #获取channel数  
    with tf.variable_scope(name) as scope:  
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])  
        b = tf.get_variable("b", shape = [featureNum])  
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideX,strideY, 1], padding = padding)  
        out = tf.nn.bias_add(featureMap, b)  
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)  
    


    
x = tf.placeholder("float", shape=[None, 28*28*1])
y_ = tf.placeholder("float", shape=[None, 10])

"""construct model"""
W_conv11 = weight_variable([5, 5, 1, 64])
b_conv11 = bias_variable([64])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv11 = tf.nn.relu(conv2d(x_image, W_conv11) + b_conv11)
h_pool1  =  maxPoolLayer(h_conv11, 2, 2, 2, 2, padding = "SAME")

W_conv21 = weight_variable([3, 3, 64, 64])
b_conv21 = bias_variable([64])
h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)
h_pool2 =  tf.reshape(maxPoolLayer(h_conv21, 2, 2, 2, 2, padding = "SAME"),[-1,7*7*64])  

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2, W_fc1) + b_fc1)

"""drop out"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

"""calculate loss function"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
tf.summary.scalar('cross_entropy',cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy',accuracy)
merged = tf.summary.merge_all()

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
"""summary"""
summary_writer = tf.summary.FileWriter('D:/tmp/mnist_logs', sess.graph)

"""set running times and prepare for the plots"""
runtimes = 1600
summary_period = 20
loss_record = np.zeros(int(runtimes/summary_period)+1)
accuracy_record = np.zeros(int(runtimes/summary_period)+1)
index = 0

sess.run(init)
for i in range(runtimes):
    batch = mnist.train.next_batch(50)
    if i%summary_period == 0:
        summary_str, train_accuracy, loss = sess.run([merged, accuracy, cross_entropy], feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g, loss %g"%(i, train_accuracy, loss))
        summary_writer.add_summary(summary_str, i)
        loss_record[index] = loss
        accuracy_record[index] = train_accuracy
        index += 1
    if i%500 == 0:
        """save model"""
        saver.save(sess, './save/model_iter', global_step = i)
        
    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
saver.save(sess, './save/model_final')
print("test accuracy %g"%accuracy.eval(feed_dict = {
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.close()

"""plot the training process"""
loss_record = loss_record[0:index]
accuracy_record = accuracy_record[0:index]
steps = np.arange(index)*summary_period

plt.figure()
plt.subplot(1,2,1)
plt.plot(steps, loss_record, marker = "*", linewidth = 3, linestyle = "--", color = "orange")
plt.title("Decrease of loss function")
plt.xlabel("Step")
plt.ylabel("Cross_entropy")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(steps, accuracy_record, linewidth = 3, linestyle = "-", color = "blue")
plt.title("Raise of accuracy")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.grid(True)

plt.show()

#tensorboard --logdir=D:/tmp/mnist_logs





