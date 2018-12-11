import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as data
mnist = data.read_data_sets("MNIST_data/", one_hot=True)
#下载并加载MNIST数据
x = tf.placeholder(tf.float32, shape = [None, 784]) #28*28*3                      
y_actual = tf.placeholder(tf.float32, shape = [None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#初始化 w

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#初始化 b
  
def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu = False)
#计算卷积，构建卷积层 cpu版 

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#定义一个函数，用于构建池化层

x_0 = tf.reshape(x, [-1,28,28,1])         
w_1 = weight_variable([5, 5, 1, 32])      
b_1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(x_0, w_1) + b_1)
#第一个卷积层
h_pool1 = max_pool(h_conv1)
#第一个池化层

w_2 = weight_variable([5, 5, 32, 64])
b_2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_2) + b_2)
#第二个卷积层
h_pool2 = max_pool(h_conv2)
#第二个池化层

w_f1 = weight_variable([7 * 7 * 64, 1024])
b_f1 = bias_variable([1024])
h_pool2flat = tf.reshape(h_pool2, [-1, 7*7*64])             
h_f1 = tf.nn.relu(tf.matmul(h_pool2flat, w_f1) + b_f1)
#全连接层
keep_prob = tf.placeholder("float") 
h_f1drop = tf.nn.dropout(h_f1, keep_prob)
#dropout优化

w_f2 = weight_variable([1024, 10])
b_f2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1drop, w_f2) + b_f2)
#softmax输出层

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))  
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
#梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                
sess=tf.InteractiveSession()                          
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:                  
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print ('step %d,accuracy of traning samples is %g'%(i,train_accuracy))
    #输出当前训练样本准确率
    train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

test_accuracy=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print ("accuracy of test samples is:%g"%test_aaccuracy)
#输出测试样本准确率
