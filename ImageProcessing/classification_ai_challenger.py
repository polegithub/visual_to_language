# coding=utf-8
import tensorflow as tf
from datasets import imnist_ai_challenger

mnist = imnist_ai_challenger.read_data_sets('./MNIST_data/', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    print("weight_variable:shape:", shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


image_length = 28

# define placeholder for inputs to network
xs = tf.placeholder(
    tf.float32, [None, image_length * image_length, 3]) / 255.   # 28x28
print('xs:')
print xs
ys = tf.placeholder(tf.float32, [None, 79])
print ys
keep_prob = tf.placeholder(tf.float32)
# print keep_prob
x_image = tf.reshape(xs, [-1, image_length, image_length, 3])
# print x_image
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 3, 32])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
# output size 28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# output size 14x14x32
h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
# output size 14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# output size 7x7x64
h_pool2 = max_pool_2x2(h_conv2)

## fc1 layer ##
W_fc1 = weight_variable([image_length * image_length / 16 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, image_length * image_length / 16 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 79])
b_fc2 = bias_variable([79])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
# merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
# writer = tf.summary.FileWriter('logs', sess.graph)  # 将训练日志写入到logs文件夹下

for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    # print batch_xs, batch_ys
    sess.run(train_step, feed_dict={
        xs: batch_xs, ys: batch_ys, keep_prob: 0.6})
    if i % 10 == 0:
        print(mnist.test.images.shape, mnist.test.labels.shape)
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
        # result = sess.run(merged, feed_dict={
        #     xs: mnist.test.images, ys: mnist.test.labels})  # 计算需要写入的日志数据
        # writer.add_summary(result, i)  # 将日志数据写入文件
