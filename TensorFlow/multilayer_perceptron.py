from tensorflow.examples.tutorials.mnist import input_data
# 使用Tensorflow提供的MNIST数据集
mnist = input_data.read_data_sets(".",one_hot=True,reshape=False)

import tensorflow as tf

# 参数Parameters,可进行调整以改善模型
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # 如果没有足够内存，可适当降低
display_step = 1

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# 隐藏层参数Hidden Layer Parameters
# n_hidden_layer 决定了神经网络隐藏层的带下。也被称作层的宽度。
n_hidden_layer = 256 # 特征的层数

# 权重和偏置项 Weights and Biases
# 层权重和偏置项的储存
weights = {
    'hidden_layer':tf.Variable(tf.random_normal([n_input,n_hidden_layer])),
    'out':tf.Variable(tf.random_normal([n_hidden_layer,n_classes]))
}
biases = {
    'hidden_layer':tf.Variable(tf.random_normal([n_hidden_layer])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

# 输入Input
x = tf.placeholder("float",[None,28,28,1])
y = tf.placeholder("float",[None,n_classes])

x_flat = tf.reshape(x,[-1,n_input]) # 将29px*28px的矩阵转换成784px*1px的单行向量

# 建立多层感知器Multilayer Perceptron
# ReLU作为隐藏层激活函数
layer_1 = tf.add(tf.matmul(x_flat,weights['hidden_layer']),biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)

# 输出层的线性激活函数
logits = tf.add(tf.matmul(layer_1,weights['out']),biases['out'])

# 定义误差值cost和优化器Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
# 这里采用了与Intro to Tensorflow lab相同的优化技术
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

## Session
# 初始化变量
init = tf.global_variables_initializer()

# 启动图
with tf.Session() as sess:
    sess.run(init)
    # 训练循环
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历所有batch
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            # 运行优化器进行反向传播，计算cost（获取loss值）
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        # 显示每步的logs
        if epoch % display_step == 0:
            c = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Epoch:",'%04d' % (epoch + 1),"cost=","{:.9f}".format(c))
    print("Optimization Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    # 如果没有足够的内存可以适当减少test_size
    test_size = 256
    print("Accuracy:",accuracy.eval({x:mnist.test.images[:test_size],y:mnist.test.labels[:test_size]}))

# Optimization Finished!
# Accuracy: 0.78515625