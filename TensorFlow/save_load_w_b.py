## 通过一个简单的例子保存weights和bias Tensors
import tensorflow as tf

# 文件保存路径
save_file = './model.ckpt'

# 两个Tensor变量：权重和偏置项
weights = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

# 用来存取Tensor变量的类
saver = tf.train.Saver()

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())

    # 显示变量和权重
    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

    # 保存模型
    saver.save(sess,save_file)

# 移除之前的权重和偏置项
tf.reset_default_graph()

# 两个变量：权重和偏置项
# 依然要创建这两个变量
weights = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

# 用来存取Tensor变量的类
saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载权重和偏置项
    saver.restore(sess,save_file)  # tf.train.Saver.restore()函数
    # 把之前保存的数据加载到weights和bias中。

    # 显示权重和偏置项
    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

# 移除之前的Tensors和运算
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

leaning_rate = 0.001
n_input = 784  # MNIST数据输入（图片尺寸：28*28）
n_classes = 10  # MNIST总计类别（数字0-9）

# 加载MNIST数据
mnist = input_data.read_data_sets('.',one_hot=True)

# 特征和标签
features = tf.placeholder(tf.float32,[None,n_input])
labels = tf.placeholder(tf.float32,[None,n_classes])

# 权重和偏置项
weights = tf.Variable(tf.random_normal([n_input,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features,weights),bias)

# 定义损失函数和优化器
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=leaning_rate).minimize(cost)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 训练模型并保存权重
import math

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()

# 启动图
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 训练循环
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)

        # 遍历所有batch
        for i in range(total_batch):
            batch_features,batch_labels = mnist.train.next_batch(batch_size)
            sess.run(
                optimizer,
                feed_dict={features:batch_features,labels:batch_labels}
            )

            # 每运行10个epoch打印一次状态
            if epoch % 10 == 0:
                valid_accuracy = sess.run(
                    accuracy,
                    feed_dict={
                        features:mnist.validation.images,
                        labels:mnist.validation.labels
                    }
                )
                print('Epoch{:<3} - Validation Accuracy:{}'.format(
                    epoch,
                    valid_accuracy
                ))

    # 保存模型
    saver.save(sess,save_file)
    print('Trained Model Saved.')

saver = tf.train.Saver()

# 加载图
with tf.Session() as sess:
    saver.restore(sess,save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={
            features:mnist.test.images,labels:mnist.test.labels
        }
    )
print('Test Accuracy:{}'.format(test_accuracy))