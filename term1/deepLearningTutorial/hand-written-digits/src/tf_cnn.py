from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)


import tensorflow as tf

# parameters

eta = 0.00001
epochs = 10
batch_size = 128

# number of samples to calculate validation and accuracy
validation_sample_size = 256

# Network Parameters

n_classes = 10
dropout_prob = 0.75


weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes])),
}

biases = {
    'bc1': tf.Variable(tf.random_normal(32)),
    'bc2': tf.Variable(tf.random_normal(64)),
    'bd1': tf.Variable(tf.random_normal(1024)),
    'out': tf.Variable(tf.random_normal(n_classes)),
}


def conv2d(x, W, b, stride=1):
    x = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, W, b, dropout):
    # layer 1: turn 28x28 image into 14x14x32
    conv1 = conv2d(x, W['wc1'], b['bc1'])
    conv1 = maxpool2d(conv1, 2)

    # layer 2: turn 14x14x32 conv layer into 7x7x64
    conv2 = conv2d(conv1, W['wc2'], b['bc2'])
    conv2 = maxpool2d(conv2, 2)

    # fully connected layer
    fc1 = tf.reshape(conv2, [-1, W['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, W), b['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # output layer
    out = tf.add(tf.matmul(fc1, W['out']), b['out'])
    return out


# Session

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)