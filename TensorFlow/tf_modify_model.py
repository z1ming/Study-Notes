import tensorflow as tf

# 移除先前的权重和偏置项
tf.reset_default_graph()

save_file = './train_model.ckpt'

# 两个Tensor变量：权重和偏置项
# 把保存的变量直接加载到已经修改过的模型会产生错误
weights = tf.Variable(tf.truncated_normal([2,3]),name='weights_0')  # 手动设置name属性
bias = tf.Variable(tf.truncated_normal([3]),name='bias_0')

saver = tf.train.Saver()

# 打印权重和偏置项的名字
print('Save Weights:{}'.format(weights.name))
print('Save Bias:{}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess,save_file)

# 移除之前的权重和偏置项
tf.reset_default_graph()

# 两个变量：权重和偏置项
bias = tf.Variable(tf.truncated_normal([3]),name='bias_0')
weights = tf.Variable(tf.truncated_normal([2,3]),name='weights_0')

saver = tf.train.Saver()

# 打印权重和偏置项的名字
print('Load Weights:{}'.format(weights.name))
print('Load Bias:{}'.format(bias.name))

with tf.Session() as sess:
    # 加载权重和偏置项-报错
    saver.restore(sess,save_file)

print('Loaded Weights and Bias successfully.')