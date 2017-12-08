import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# hyperpatameters
lr = 0.001             # learning rate
training_iters = 100000
batch_size = 128
display_step = 20

n_inputs = 28          # MNIST data input    image shape: 28 * 28
n_step = 28            # time step
n_hidden_unis = 128    # neurons in hidden layer
n_class = 10           # MNIST class (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_step, n_inputs])
y = tf.placeholder(tf.float32, [None, n_class])

# define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_class]))
}
biases = {
    # (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis,])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_class,]))
}

def RNN_lstm(X, weights, biases):
    # hidden layer for input to cell
    # X(128 batch, 28 step, 28 inputs)
    # ==> (128*28, 28)
    X = tf.reshape(X, [-1, n_inputs])
    # ==> (128batch, 28step, 128 n_hidden_units)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_unis])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts: (c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    results = tf.matmul(states[1], weights['out']) + biases['out']

#    outputs = tf.unpack(tf.transpose(outputs, [1,0,2]))
#    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

def RNN_gru(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_unis])

    gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden_unis)
    _init_state = gru_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(gru_cell, X_in, initial_state=_init_state, time_major=False)
    results = tf.matmul(states, weights['out']) + biases['out']

    return results

pred = RNN_lstm(x, weights, biases)
#pred = RNN_gru(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_inputs])
        sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})
        if step % display_step == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
        step += 1