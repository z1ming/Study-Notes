import tensorflow as tf

hidden_layer_weights = [
    [0.1,0.2,0.4],
    [0.4,0.6,0.6],
    [0.5,0.9,0.1],
    [0.8,0.2,0.8]]
out_weights = [
    [0.1,0.6],
    [0.2,0.1],
    [0.7,0.9]]

weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)
]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))
]

# Input
features = tf.Variable([[0.0,2.0,3.0,4.0],[0.1,0.2,0.3,0.4],[11.0,12.0,13.0,14.0]])

# Create Model with Dropout
keep_prob = tf.placeholder(tf.float32)
layer = tf.add(tf.matmul(features,weights[0]),biases[0])
layer = tf.nn.relu(layer)
layer = tf.nn.dropout(layer,keep_prob)

logits = tf.add(tf.matmul(layer,weights[1]),biases[1])

# Print logtis from a session
# 初始化
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits,feed_dict={keep_prob:0.5}))# 在此设置keep_prob参数也行