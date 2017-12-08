import tensorflow as tf

## save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "my_net/save_net.ckpt")
#     print("save to path:", save_path)
## save to file end



# restore variables
# redefine the same shape and same type for your variables
W = tf.Variable(tf.zeros([2,3]), dtype=tf.float32, name='weights')
b = tf.Variable(tf.zeros([1,3]), dtype=tf.float32, name='biases')
# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("w:", sess.run(W))
    print("b:", sess.run(b))

