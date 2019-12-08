import tensorflow as tf

def run():
    output = None
    logits_data = [2.0,1.0,0.1]
    logits = tf.placeholder(tf.float32)  # 使用非常量

    # Calculate the softamx of the logits

    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        output = sess.run(softmax,feed_dict={logits:logits_data})

    return output