import tensorflow as tf

def get_weights(n_features,n_labels):
    """
    Return Tensorflow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: Tensorflow weights
    """
    # Return weights
    # 使用tf.Variable()返回可修改的权重，采用随机化，
    # 因此用正态分布tf.truncated_normal()
    return tf.Variable(tf.truncated_normal((n_features,n_labels)))

def get_biases(n_labels):
    """
    Return Tensorflow bias
    :param n_labels: Number of labels
    :return: Tensorflow bias
    """
    # 因为权重已经被随机化来帮助模型不被卡住，因此不需要再随机化偏差了，设为0.
    return tf.Variable(tf.zeros(n_labels))

def linear(input,w,b):
    """
    Return linear function in Tensorflow
    :param input: Tensorflow input
    :param w: Tensorflow weights
    :param b: Tensorflow biases
    :return: TensorFlow linear function
    """
    # 使用tf.matmul()函数进行矩阵乘法
    return tf.add(tf.matmul(input,w),b)