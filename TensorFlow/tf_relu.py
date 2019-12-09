import tensorflow as tf

output = None
hidden_layer_weight = [
    [0.1,0.2,0.4],
    [0.4,0.6,0.6],
    [0.5,0.9,0.1],
    [0.8,0.2,0.8]]
out_weights = [
    [0.1,0.6],
    [0.2,0.1],
    [0.7,0.9]]

weights = [
    tf.Variable(hidden_layer_weight),
    tf.Variable(out_weights)
]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))
]

# Input
features = tf.Variable([[1.0,2.0,3.0,4.0],[-1.0,-2.0,-3.0,-4.0],[11.0,12.0,13.0,14.0]])

# Create Model
hidden_layer = tf.nn.relu(tf.add(tf.matmul(features,weights[0]),biases[0]))
logits = tf.add(tf.matmul(hidden_layer,weights[1]),biases[1])

# Print session results
# 初始化
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits))
