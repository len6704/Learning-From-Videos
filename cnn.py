
import tensorflow as tf
import sys
import pickle as pickle
import numpy as np
import os

sys.path.append("/Users/len/Desktop/MLCV")
from data_utils import load_CIFAR_batch
X,Y = load_CIFAR_batch("/Users/len/Desktop/MLCV/cifar-10-batches-py/data_batch_1")
#Using CIFAR dataset as initial testing data

# Training Parameters
learning_rate = 0.001
num_steps = 200
num_epochs = 10
display_step = 1
total_data = 10000

# Network Parameters
input_dimension = 3072 # 32*32*3
num_classes = 10 # 
dropout = 0.75 # Dropout, probability to keep units
num_train = 1000
num_test = 300
batch_size = 80
#mini_data['X_train'] = X[:num_train]
#mini_data['Y_train'] = Y[:num_train]
label_matrix = np.zeros([total_data,num_classes])
for i in range(total_data):
    label = Y[i]
    label_matrix[i,label] = 1
Y = label_matrix
#print (mini_data['Y_train'][:3])
#print (label_matrix[:3])
#mini_data['X_val'] = data['X'][0:num_train]
#mini_data['Y_val'] = data['Y'][0:num_train]
X_train = X[:num_train,:,:,:]
Y_train = Y[:num_train,:]
X_test  = X[num_train+1 : num_train+num_test+1,:,:,:]
Y_test  = Y[num_train+1 : num_train+num_test+1,:]
#X_test =  np.reshape(X_test,[num_test,-1])
print(X.shape)



# tf Graph input
#X = tf.placeholder(tf.float32, [None, input_dimension])
#Y = tf.placeholder(tf.float32, [None, num_classes])
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    # Convolution Layer
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    #print (conv1.shape)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # Convolution Layer
    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    # Convolution Layer
    conv3 = conv2d(conv2, weights['w3'], biases['b3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(conv3, [-1, weights['w4'].get_shape().as_list()[0]])
    fc1 = tf.reshape(conv3, [-1, weights['w4'].shape[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w4']), biases['b4'])
    fc1 = tf.nn.relu(fc1)
    
    fc2 = tf.add(tf.matmul(fc1, weights['w5']), biases['b5'])
    return fc2

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 32 outputs
    'w1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 32 outputs
    'w2': tf.Variable(tf.random_normal([5, 5, 32, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 5*5*64 inputs, 64 outputs
    'w4': tf.Variable(tf.random_normal([1*1*64, 64])),
    # 64 inputs, 48 outputs (class prediction)
    'w5': tf.Variable(tf.random_normal([64, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([32])),
    'b2': tf.Variable(tf.random_normal([32])),
    'b3': tf.Variable(tf.random_normal([64])),
    'b4': tf.Variable(tf.random_normal([64])),
    'b5': tf.Variable(tf.random_normal([num_classes]))
}


''' Testing architecture. Can ignore this part
# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    #x = tf.reshape(x, shape=[-1, 32, 32, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['w4'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w4']), biases['b4'])
    fc1 = tf.nn.relu(fc1)
    
    fc2 = tf.add(tf.matmul(fc1, weights['w5']), biases['b5'])
    
    return fc2

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'w1': tf.Variable(tf.random_normal([7, 7, 3, 64])),
    # fully connected, 5*5*64 inputs, 64 outputs
    'w4': tf.Variable(tf.random_normal([13*13*64, 100])),
    # 64 inputs, 48 outputs (class prediction)
    'w5': tf.Variable(tf.random_normal([100, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([64])),
    'b4': tf.Variable(tf.random_normal([100])),
    'b5': tf.Variable(tf.random_normal([num_classes]))
}
'''





# Construct model
logits = conv_net(X, weights, biases, keep_prob)

prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    #print (weights['w1'].eval())
    print ("----------")
    for epoch in range(1,num_epochs+1):
        print("Epoch:" + str(epoch))
        for step in range(1, num_steps+1):
            batch_mask = np.random.choice(num_train,batch_size)     
            batch_x =  X_train[batch_mask,:,:,:]
            batch_y =  Y_train[batch_mask,:]
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    
    print("Optimization Finished!")
    #batch_size = num_test
    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_test, Y: Y_test, keep_prob: 1.0})  
    #print (weights['w1'].eval())
    print("Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Testing Accuracy= " + \
                  "{:.3f}".format(acc))


