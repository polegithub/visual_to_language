# coding=utf-8
import tensorflow as tf
import numpy as np
from datasets import data_sets_ai_challenger

data_sets = data_sets_ai_challenger.read_data_sets(
    './MNIST_data/', one_hot=True)
scene_classes_csv = data_sets_ai_challenger.load_ai_scene_classes_csv()


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    print("预测场景：真实场景")
    index_tensor_pre = tf.argmax(y_pre, axis=1)
    indexes_pre = sess.run(index_tensor_pre)
    indexes_pre_to_list = indexes_pre.tolist()

    index_tensor_ys = tf.argmax(v_ys, axis=1)
    indexes_ys = sess.run(index_tensor_ys)
    indexes_ys_to_list = indexes_ys.tolist()

    for index in range(0, len(indexes_ys_to_list)):
        index_ys = indexes_ys_to_list[index]
        index_pre = indexes_pre_to_list[index]
        print(scene_classes_csv[index_pre] +
              ":" + scene_classes_csv[index_ys])
    correct_prediction = tf.equal(index_tensor_pre, index_tensor_ys)
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
with tf.name_scope('input_layer'):  # 输入层。将这两个变量放到input_layer作用域下，tensorboard会把他们放在一个图形里面
    xs = tf.placeholder(
        tf.float32, [None, image_length * image_length, 3], name='x_input') / 255.   # 28x28
    ys = tf.placeholder(tf.float32, [None, 79], name='y_input')
    keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, image_length, image_length, 3])
# print(x_image.shape)  # [n_samples, 28,28,1]

with tf.name_scope('conv1_layer'):
    ## conv1 layer ##
    with tf.name_scope('weight'):
        # patch 5x5, in size 1, out size 32
        W_conv1 = weight_variable([5, 5, 3, 32])
        tf.summary.histogram('conv1_layer/weight', W_conv1)
    with tf.name_scope('bias'):
        b_conv1 = bias_variable([32])
        tf.summary.histogram('conv1_layer/bias', b_conv1)
    with tf.name_scope('h_conv'):
        # output size 28x28x32
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram('conv1_layer/h_conv', h_conv1)
    with tf.name_scope('h_pool'):
        # output size 14x14x32
        h_pool1 = max_pool_2x2(h_conv1)
        tf.summary.histogram('conv1_layer/h_pool', h_pool1)

## conv2 layer ##
with tf.name_scope('conv2_layer'):
    ## conv1 layer ##
    with tf.name_scope('weight'):
        # patch 5x5, in size 32, out size 64
        W_conv2 = weight_variable([5, 5, 32, 64])
        tf.summary.histogram('conv2_layer/weight', W_conv2)
    with tf.name_scope('bias'):
        b_conv2 = bias_variable([64])
        tf.summary.histogram('conv2_layer/bias', b_conv2)
    with tf.name_scope('h_conv'):
        # output size 14x14x64
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        tf.summary.histogram('conv2_layer/h_conv', h_conv2)
    with tf.name_scope('h_pool'):
        # output size 7x7x64
        h_pool2 = max_pool_2x2(h_conv2)
        tf.summary.histogram('conv2_layer/h_pool', h_pool2)

## fc1 layer ##
with tf.name_scope('fc1_layer'):
    with tf.name_scope('weight'):  # 权重
        W_fc1 = weight_variable([image_length * image_length / 16 * 64, 1024])
        tf.summary.histogram('fc1_layer/weight', W_fc1)
    with tf.name_scope('bias'):  # 偏置
        b_fc1 = bias_variable([1024])
        tf.summary.histogram('fc1_layer/bias', b_fc1)
    with tf.name_scope('h_pool_flat'):
        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(
            h_pool2, [-1, image_length * image_length / 16 * 64])
        tf.summary.histogram('fc1_layer/h_pool_flat', h_pool2_flat)
    with tf.name_scope('h_fc'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        tf.summary.histogram('fc1_layer/h_fc', h_fc1)
    with tf.name_scope('h_fc_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        tf.summary.histogram('fc1_layer/h_fc_drop', h_fc1_drop)

## fc2 layer ##
with tf.name_scope('fc2_layer'):
    with tf.name_scope('weight'):  # 权重
        W_fc2 = weight_variable([1024, 79])
        tf.summary.histogram('fc2_layer/weight', W_fc2)
    with tf.name_scope('bias'):  # 偏置
        b_fc2 = bias_variable([79])
        tf.summary.histogram('fc2_layer/bias', b_fc2)
    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        tf.summary.histogram('fc2_layer/prediction', prediction)


# the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))       # loss
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):  # 训练过程
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
writer = tf.summary.FileWriter('logs', sess.graph)  # 将训练日志写入到logs文件夹下
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

for i in range(1000):
    batch_xs, batch_ys = data_sets.train.next_batch(100)
    # print batch_xs, batch_ys
    sess.run(train_step, feed_dict={
        xs: batch_xs, ys: batch_ys, keep_prob: 0.6})
    if i % 50 == 0:
        print(compute_accuracy(
            data_sets.test.images, data_sets.test.labels))
        result = sess.run(merged, feed_dict={
            xs: data_sets.test.images, ys: data_sets.test.labels, keep_prob: 0.6})  # 计算需要写入的日志数据
        writer.add_summary(result, i)  # 将日志数据写入文件
        saver.save(sess, 'model/model.ckpt')
